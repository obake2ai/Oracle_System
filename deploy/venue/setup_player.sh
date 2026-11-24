#!/usr/bin/env bash
# 会場の小型 PC (Ubuntu デスクトップ、OptiPlex 7060 を想定) に Organs のプレイヤーを据える。
# ログイン時に Chrome が全画面で配信を開き、回線が切れたら player/fallback.mp4 に切り替わる。
#   bash deploy/venue/setup_player.sh <POD_ID>        設置 (例: p6wry0zxb382lc)
#   bash deploy/venue/setup_player.sh remove          自動起動を外す (ファイルは残す)
# 2026-11-24 作成。実機では未実行 (12/1 に OptiPlex で確認する)。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PLAYER=$ROOT/player
PORT=8088
UNIT=$HOME/.config/systemd/user/organs-player-http.service
DESKTOP=$HOME/.config/autostart/organs-player.desktop

if [ "${1:-}" = "remove" ]; then
  systemctl --user disable --now organs-player-http.service 2>/dev/null || true
  rm -f "$UNIT" "$DESKTOP"
  echo "自動起動を外しました"; exit 0
fi

POD_ID=${1:?POD_ID を指定してください}

BROWSER=""
for b in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$b" >/dev/null 2>&1; then BROWSER=$(command -v "$b"); break; fi
done
[ -n "$BROWSER" ] || { echo "Chrome / Chromium が見つかりません。先に入れてください"; exit 1; }
echo "browser: $BROWSER / session: ${XDG_SESSION_TYPE:-unknown} / desktop: ${XDG_CURRENT_DESKTOP:-unknown}"

# 配信 URL を書き込む (Git の config.js は POD_ID のままにしておき、実機でだけ書き換える)
sed -i -E "s#https://[A-Za-z0-9_]+-8000\.proxy\.runpod\.net#https://${POD_ID}-8000.proxy.runpod.net#" "$PLAYER/config.js"
grep -n "streamUrl" "$PLAYER/config.js"
[ -f "$PLAYER/fallback.mp4" ] || echo "注意: $PLAYER/fallback.mp4 がまだありません (回線断の時は黒になります)"

# プレイヤーを手元の HTTP で配る (file:// より Chrome の制限に当たりにくい)
mkdir -p "$(dirname "$UNIT")" "$(dirname "$DESKTOP")"
cat > "$UNIT" <<EOF
[Unit]
Description=Organs venue player (local HTTP)

[Service]
WorkingDirectory=$PLAYER
ExecStart=/usr/bin/python3 -m http.server $PORT --bind 127.0.0.1
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now organs-player-http.service

# ログイン時に全画面で開く
cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Organs player
Exec=$BROWSER --kiosk --incognito --noerrdialogs --disable-session-crashed-bubble --autoplay-policy=no-user-gesture-required http://127.0.0.1:$PORT/index.html
X-GNOME-Autostart-enabled=true
EOF

# 画面が消えないようにする (GNOME の場合)
if command -v gsettings >/dev/null 2>&1; then
  gsettings set org.gnome.desktop.session idle-delay 0 || true
  gsettings set org.gnome.desktop.screensaver lock-enabled false || true
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' || true
fi

echo "設置しました。確認: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$PORT/index.html"
echo "残りの手作業: 自動ログインの設定 / BIOS の AC Power Recovery=On / 他の展示用の cron (crontab -l) の停止"
