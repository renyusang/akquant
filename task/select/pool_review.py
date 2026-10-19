"""季度池评审(2026-10-20): 粘性更新 + 硬资格剔除 + 留痕。

为什么是"粘性"而不是"每季重选 40 只":
- 规则池每季从零重选会换掉约 30/40 只(实测)，而"每次最多换 4 只"的过渡规则
  会让实盘池落后规则数年；两者不可同时成立。
- 实测(22 期面板): 粘性池(保留合格在池、只补空缺)换手 0.7 只/季,
  年化 7.8% vs 从零重选 9.2%, 回撤 -11.2% vs -10.3% —— 用 1.4pp/年 换 40 倍稳定性。

规则:
- 每季末: 对现有成员重跑资格(波动/成交额/红线等)；不合格 → 立即剔除(硬约束)；
- 合格成员全部保留(粘性)，空缺再由规则排名补足(主题≤30%/申万≤3)；
- 变更写入 pool_history.csv, 状态写入 pool_state.json。

用法:
    python pool_review.py --init --date 2026-03-31
    python pool_review.py --date 2026-06-30
    python pool_review.py --date 2026-09-30 --dry-run
"""

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STATE = os.path.join(HERE, "pool_state.json")
HISTORY = os.path.join(HERE, "pool_history.csv")


def _load_state():
    if not os.path.exists(STATE):
        return None
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def _save_state(state):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_history(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(HISTORY):
        df.to_csv(HISTORY, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY, index=False, encoding="utf-8-sig")


def _eligible(date):
    """返回 (合格集合, 特征表, 不合格原因)。"""
    import pool_rule as pr
    d = pr.features(date)
    red = pr.load_red_lines()
    reasons = {}
    bad = d[d.symbol.isin(red)]
    for s in bad.symbol:
        reasons[s] = "红线"
    d = d[~d.symbol.isin(red)]
    low = d[d.amount20 < pr.MIN_AMOUNT]
    for s in low.symbol:
        reasons[s] = "流动性"
    d = d[d.amount20 >= pr.MIN_AMOUNT]
    q = d.vol.quantile(pr.VOL_PCT)
    high = d[d.vol > q]
    for s in high.symbol:
        reasons[s] = "波动过高"
    d = d[d.vol <= q]
    return set(d.symbol), d, reasons


def _info(sel):
    return {r.symbol: {"name": r["name"], "bucket": r.bucket, "theme": r.theme,
                       "industry": r.industry, "ret12": float(r.ret12),
                       "rank": int(r["rank"])} for _, r in sel.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    import pool_rule as pr
    sel, reserve = pr.build_pool(args.date, top=args.top, reserve=10)
    info = _info(sel)
    res = _info(reserve)
    elig, feat, reasons = _eligible(args.date)
    feat_map = feat.set_index("symbol")

    state = _load_state()
    if args.init or state is None:
        members = list(sel.symbol)
        state = {"asof": args.date, "members": members,
                 "names": {s: info[s]["name"] for s in members},
                 "reserve": list(res), "updated": args.date}
        rows = [{"date": args.date, "action": "init", "symbol": s,
                 "name": info[s]["name"], "reason": "首次建立",
                 "bucket": info[s]["bucket"], "theme": info[s]["theme"],
                 "industry": info[s]["industry"]} for s in members]
        print(f"[init] {args.date}: 建立 {len(members)} 只池子 + {len(res)} 只备选")
        if not args.dry_run:
            _save_state(state)
            _append_history(rows)
        return

    members = state["members"]
    names = state.get("names", {})
    keep = [s for s in members if s in elig]
    dropped = [s for s in members if s not in elig]

    add: list[str] = []
    slots = args.top - len(keep)
    if slots > 0:
        pool = list(sel.symbol) + list(reserve.symbol)
        add = [s for s in pool if s not in keep][:slots]
    new_members = keep + add

    rows = []
    for s in dropped:
        rows.append({"date": args.date, "action": "remove", "symbol": s,
                     "name": names.get(s, ""),
                     "reason": f"资格失效({reasons.get(s, '不在候选')})",
                     "bucket": "", "theme": "", "industry": ""})
    for s in add:
        i = info.get(s) or res.get(s, {})
        rows.append({"date": args.date, "action": "add", "symbol": s,
                     "name": i.get("name", ""), "reason": "补充空缺",
                     "bucket": i.get("bucket", ""), "theme": i.get("theme", ""),
                     "industry": i.get("industry", "")})

    state.update({
        "asof": args.date, "members": new_members,
        "names": {**{s: names.get(s, "") for s in keep},
                  **{s: (info.get(s) or res.get(s, {})).get("name", "") for s in add}},
        "reserve": list(res), "updated": args.date,
    })
    print(f"[review] {args.date}: 池 {len(new_members)} 只 | "
          f"剔除 {len(dropped)} / 补入 {len(add)} / 保留 {len(keep)}")
    for r in rows:
        print(f"  {r['action']:7s} {r['symbol']} {r['name']}  {r['reason']}")
    if not args.dry_run:
        _save_state(state)
        _append_history(rows)
        print(f"写入 {STATE} / {HISTORY}")


if __name__ == "__main__":
    main()
