import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { mkdir } from 'node:fs/promises';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import ffmpegPath from 'ffmpeg-static';

const ROOT = dirname(fileURLToPath(import.meta.url));
const OUT = join(ROOT, 'exports');
const HTML = 'day6-animation.html';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const FPS = 30;
const DURATION = 56;
const HOLD = 1.0;
const FRAMES = Math.round((DURATION + HOLD) * FPS);
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
};

function runFfmpeg(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(ffmpegPath, args, { stdio: ['pipe', 'inherit', 'inherit'] });
    proc.on('error', reject);
    proc.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited ${code}\n${args.join(' ')}`));
    });
  });
}

async function serve() {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1');
    const file = url.pathname === '/' ? HTML : url.pathname.slice(1);
    try {
      const buf = await readFile(join(ROOT, file));
      res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
      res.end(buf);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return server;
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const server = await serve();
  const port = server.address().port;
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: [
      '--hide-scrollbars',
      '--disable-font-subpixel-positioning',
      '--font-render-hinting=none',
      `--window-size=3840,2160`,
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 3840, height: 2160, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:${port}/${HTML}?record`, {
    waitUntil: 'networkidle0',
    timeout: 60000,
  });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => document.getElementById('c')?.width === 3840);

  const master = join(OUT, 'day6-4k-master.mp4');
  const twitter = join(OUT, 'day6-twitter.mp4');
  const gif = join(OUT, 'day6.gif');

  const enc = spawn(ffmpegPath, [
    '-y',
    '-f', 'image2pipe',
    '-framerate', String(FPS),
    '-vcodec', 'mjpeg',
    '-i', 'pipe:0',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-tune', 'animation',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-profile:v', 'high',
    '-r', String(FPS),
    '-g', String(FPS),
    '-bf', '0',
    '-movflags', '+faststart',
    master,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });

  const done = new Promise((resolve, reject) => {
    enc.on('error', reject);
    enc.on('close', code => (code === 0 ? resolve() : reject(new Error(`master encode exited ${code}`))));
  });

  for (let i = 0; i < FRAMES; i++) {
    const dataUrl = await page.evaluate(frame => window.__drawRecordFrame(frame), i);
    const buf = Buffer.from(dataUrl.split(',')[1], 'base64');
    if (!enc.stdin.write(buf)) {
      await new Promise(r => enc.stdin.once('drain', r));
    }
    if (i % 30 === 0) console.log(`captured ${i}/${FRAMES}  jpeg=${(buf.length / 1024).toFixed(0)}KB`);
  }
  enc.stdin.end();
  await done;
  await browser.close();
  server.close();
  console.log('4K master written');

  await runFfmpeg([
    '-y', '-i', master,
    '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
    '-c:v', 'libx264',
    '-preset', 'slow',
    '-tune', 'animation',
    '-profile:v', 'high',
    '-level', '4.2',
    '-pix_fmt', 'yuv420p',
    '-vf', 'scale=1920:1080:flags=lanczos,format=yuv420p',
    '-b:v', '12M',
    '-maxrate', '12M',
    '-bufsize', '24M',
    '-r', '30',
    '-g', '30',
    '-bf', '2',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-shortest',
    '-movflags', '+faststart',
    twitter,
  ]);
  console.log('Twitter MP4 written');

  await runFfmpeg([
    '-y', '-i', master,
    '-vf', [
      'fps=12',
      'scale=960:540:flags=lanczos',
      'split[s0][s1]',
      '[s0]palettegen=max_colors=256:stats_mode=full[p]',
      '[s1][p]paletteuse=dither=floyd_steinberg',
    ].join(','),
    '-loop', '0',
    gif,
  ]);
  console.log('GIF written');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
