// 会場プレイヤーの設定。RunPod の Pod ID が決まったら streamUrl を書き換える。
// URL の ?src=...&debug=1 で一時的に上書きできる。
window.ORGANS_PLAYER = {
  // RunPod の HTTP プロキシ経由 (HTTPS 443 で届くので学内ファイアウォールを通りやすい)
  streamUrl: "https://POD_ID-8000.proxy.runpod.net/stream.m3u8",
  // 回線が落ちた時に流す、同じシステムで事前に録った長尺映像 (このフォルダに置く)
  fallbackUrl: "fallback.mp4",
  stallSeconds: 12,        // ライブの再生がこの秒数進まなければ予備映像に切り替える
  retrySeconds: 20,        // 予備映像を流している間、この間隔でライブを再接続する
  healthySeconds: 6,       // 再接続したライブがこの秒数安定して進んだらライブに戻す
  dailyReloadAt: "06:00"   // 毎日この時刻にページを読み直す (長時間稼働の保険)。空文字で無効
};
