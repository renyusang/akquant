#!/usr/bin/env python3
"""
一键服务器初始化脚本 — 三步完成报告部署环境配置。

步骤:
  1. 安装 SSH 公钥 (免密登录)
  2. 安装/配置 Nginx (监听 8888 端口 + 静态文件服务)
  3. 创建报告目录 + 写入初始 index.html

用法:
  python setup_server.py           # 交互式运行全部步骤
  python setup_server.py --check   # 仅检查服务器环境
"""

import os
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_creds():
    creds_path = os.path.join(TASK_DIR, "report_deploy.md")
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"{creds_path} not found")
    with open(creds_path) as f:
        content = f.read()
    import re
    ip_match = re.search(r'ip是([\d.]+)', content)
    user_match = re.search(r'用户名是([a-zA-Z0-9]+)', content)
    pw_match = re.search(r'passwd 是([\x21-\x7e]+)', content)  # printable ASCII
    if not all([ip_match, user_match, pw_match]):
        raise ValueError("Cannot parse credentials from report_deploy.md")
    return ip_match.group(1), user_match.group(1), pw_match.group(1)


def _ssh_exec(client, cmd, desc=""):
    """Execute command on remote server, print output."""
    if desc:
        print(f"  [{desc}]")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        for line in out.split("\n"):
            print(f"    {line}")
    if err:
        for line in err.split("\n"):
            print(f"    [stderr] {line}")
    return out, err


def step1_ssh_key(client):
    """Install SSH public key for passwordless login."""
    print("\n" + "=" * 50)
    print("Step 1: SSH Key Setup")
    print("=" * 50)

    for key_name in ["id_ed25519.pub", "id_rsa.pub"]:
        pubkey_path = os.path.expanduser(f"~/.ssh/{key_name}")
        if os.path.exists(pubkey_path):
            break
    else:
        print("No SSH key found. Generating rsa 4096...")
        os.system("ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N '' -q")
        pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")

    with open(pubkey_path) as f:
        pubkey = f.read().strip()
    print(f"  Key: {pubkey_path}")

    cmd = (
        'mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
        f'echo "{pubkey}" >> ~/.ssh/authorized_keys && '
        'chmod 600 ~/.ssh/authorized_keys && '
        'echo "KEY_INSTALLED"'
    )
    out, err = _ssh_exec(client, cmd, "Installing public key")
    if "KEY_INSTALLED" in out:
        print("  ✓ SSH key installed")
    else:
        print("  ⚠ Key may already exist, continuing...")


def step2_nginx(client):
    """Install and configure nginx on server."""
    print("\n" + "=" * 50)
    print("Step 2: Nginx Setup")
    print("=" * 50)

    # Check if nginx is already installed
    out, _ = _ssh_exec(client, "which nginx 2>/dev/null && nginx -v 2>&1 || echo 'NO_NGINX'")
    has_nginx = "NO_NGINX" not in out

    if not has_nginx:
        print("  Installing nginx...")
        out, err = _ssh_exec(
            client,
            "sudo apt-get update -qq && sudo apt-get install -y -qq nginx 2>&1",
            "apt-get install nginx"
        )
        # Verify
        out2, _ = _ssh_exec(client, "which nginx && nginx -v 2>&1")
        if "nginx" in out2:
            print("  ✓ Nginx installed")
        else:
            print("  ✗ Nginx installation may have failed, check output above")
            return False
    else:
        print("  ✓ Nginx already installed")

    # Read nginx config template and upload it
    nginx_conf_path = os.path.join(TASK_DIR, "nginx-kalman-reports.conf")
    if not os.path.exists(nginx_conf_path):
        print("  ✗ nginx-kalman-reports.conf not found!")
        return False

    with open(nginx_conf_path) as f:
        nginx_conf = f.read()

    # Escape for echo/sed
    import base64
    conf_b64 = base64.b64encode(nginx_conf.encode()).decode()

    # Write config via base64 (avoids escaping issues)
    cmd = (
        f'echo "{conf_b64}" | base64 -d | '
        'sudo tee /etc/nginx/sites-available/kalman-reports > /dev/null && '
        'echo "CONF_WRITTEN"'
    )
    out, _ = _ssh_exec(client, cmd, "Writing nginx config")
    if "CONF_WRITTEN" not in out:
        print("  ✗ Failed to write nginx config")
        return False
    print("  ✓ Nginx config written: /etc/nginx/sites-available/kalman-reports")

    # Enable site (symlink)
    _ssh_exec(
        client,
        "sudo ln -sf /etc/nginx/sites-available/kalman-reports "
        "/etc/nginx/sites-enabled/kalman-reports && "
        "echo 'SITE_ENABLED'",
        "Enabling site"
    )

    # Remove default site if it conflicts on port 80
    _ssh_exec(
        client,
        "sudo rm -f /etc/nginx/sites-enabled/default",
        "Removing default site (if exists)"
    )

    # Create log directory
    _ssh_exec(
        client,
        "sudo mkdir -p /var/log/nginx && sudo touch /var/log/nginx/kalman-reports-access.log "
        "/var/log/nginx/kalman-reports-error.log && "
        "sudo chown -R www-data:adm /var/log/nginx/kalman-reports-*.log 2>/dev/null; "
        "echo 'LOG_OK'",
        "Setting up log files"
    )

    # Test config
    out, err = _ssh_exec(client, "sudo nginx -t 2>&1", "Testing nginx config")
    if "successful" in out.lower() or "ok" in out.lower():
        print("  ✓ Nginx config test passed")
    else:
        print(f"  ⚠ Config test output: {out[:200]}")

    # Reload/start nginx
    _ssh_exec(client, "sudo systemctl reload nginx 2>/dev/null || sudo nginx -s reload 2>/dev/null || sudo systemctl start nginx", "Reloading nginx")
    print("  ✓ Nginx reloaded")

    return True


def step3_dirs(client):
    """Create report directories and initial index.html."""
    print("\n" + "=" * 50)
    print("Step 3: Report Directories")
    print("=" * 50)

    dirs = "/var/www/reports /var/www/reports/live /var/www/reports/backtest/single /var/www/reports/backtest/portfolio"
    _ssh_exec(client, f"sudo mkdir -p {dirs} && sudo chown -R $USER:$USER /var/www/reports && echo 'DIRS_OK'", "Creating directories")
    print("  ✓ Report directories created: /var/www/reports/")

    # Create a placeholder index.html
    placeholder = '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
    placeholder += '<title>AKQuant Reports</title></head><body>'
    placeholder += '<h1>AKQuant Kalman Reports</h1>'
    placeholder += '<p>Reports coming soon. Run deploy.sh to upload.</p>'
    placeholder += '</body></html>'

    import base64
    ph_b64 = base64.b64encode(placeholder.encode()).decode()
    _ssh_exec(
        client,
        f'echo "{ph_b64}" | base64 -d > /var/www/reports/index.html && echo "INDEX_OK"',
        "Creating placeholder index.html"
    )
    print("  ✓ Placeholder index.html created")

    # Verify port is listening
    out, _ = _ssh_exec(client, "ss -tlnp 2>/dev/null | grep 8888 || ss -tln 2>/dev/null | grep 8888 || echo 'PORT_NOT_LISTENING'", "Checking port 8888")
    if "PORT_NOT_LISTENING" in out:
        print("  ⚠ Port 8888 not listening yet. Manually check:")
        print("    sudo systemctl status nginx")
    else:
        print("  ✓ Port 8888 is listening!")
        print(f"    {out}")


def check_only(client):
    """Check server environment without making changes."""
    print("\n" + "=" * 50)
    print("Server Environment Check")
    print("=" * 50)

    checks = {
        "OS": "uname -a",
        "Disk": "df -h / | tail -1",
        "Memory": "free -h | head -2",
        "Nginx": "which nginx 2>/dev/null && nginx -v 2>&1 || echo 'NO_NGINX'",
        "Systemd": "which systemctl 2>/dev/null && echo 'HAS_SYSTEMD' || echo 'NO_SYSTEMD'",
        "Python3": "python3 --version 2>&1 || echo 'NO_PYTHON3'",
        "Port_8888": "ss -tlnp 2>/dev/null | grep 8888 || ss -tln 2>/dev/null | grep 8888 || echo 'PORT_8888_FREE'",
        "Nginx_sites": "ls -la /etc/nginx/sites-enabled/ 2>/dev/null || echo 'NO_SITES'",
        "www_dir": "ls -la /var/www/ 2>/dev/null || echo 'NO_VAR_WWW'",
    }

    for name, cmd in checks.items():
        print(f"\n  --- {name} ---")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            for line in out.split("\n"):
                print(f"    {line}")
        if err:
            for line in err.split("\n"):
                print(f"    [err] {line}")

    print()


def main():
    import paramiko

    host, user, password = _read_creds()
    print(f"Target: {user}@{host}:8888")

    check_only_mode = "--check" in sys.argv

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=user, password=password, timeout=15)
        print("✓ SSH connected (password auth)\n")

        if check_only_mode:
            check_only(client)
        else:
            step1_ssh_key(client)
            step2_nginx(client)
            step3_dirs(client)

            print("\n" + "=" * 50)
            print("✓ Setup complete!")
            print(f"  Visit: http://{host}:8888/")
            print("=" * 50)

            # Test key-based login
            print("\nTesting key-based SSH login...")
            client.close()
            import subprocess
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=accept-new",
                 "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 f"{user}@{host}", "echo KEY_LOGIN_OK"],
                capture_output=True, text=True, timeout=15
            )
            if "KEY_LOGIN_OK" in result.stdout:
                print("✓ Key-based SSH login working! You can now run deploy.sh")
            else:
                print(f"⚠ Key login test: {result.stdout.strip()} {result.stderr.strip()}")
                print("  You may need to manually add the SSH key or check server config.")

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if server is running: ping {host}")
        print("  2. Check if SSH port 22 is open")
        print("  3. Check credentials in report_deploy.md")
        sys.exit(1)

    client.close()


if __name__ == "__main__":
    main()
