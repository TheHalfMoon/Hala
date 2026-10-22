/* H61 batch screenshot capture in a real headless browser.
 *
 * One browser process is started and every page is captured through the
 * DevTools protocol, so a few thousand first-party scenes do not cost one
 * browser launch each. The capture is the genuine rendered pixels of the
 * genuine installed browser; nothing is drawn by this script.
 *
 * Usage: node capture_browser.js <browser> <output-dir> <manifest-file>
 * The manifest is one page path per line, so the corpus size is not bounded
 * by the operating system command line length.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');

const [browser, outputDir, manifest] = process.argv.slice(2);
if (!browser || !outputDir || !manifest) {
  console.error('usage: capture_browser.js <browser> <output-dir> <manifest>');
  process.exit(2);
}
const pages = fs.readFileSync(manifest, 'utf8').split('\n').map((line) => line.trim())
  .filter((line) => line.length > 0);
if (!pages.length) {
  console.error('H61_CAPTURE_FAILED the manifest holds no page');
  process.exit(2);
}
const PORT = 9222 + (process.pid % 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'h61-capture-'));
fs.mkdirSync(outputDir, { recursive: true });

const child = spawn(browser, [
  '--headless=new',
  '--disable-gpu',
  '--no-sandbox',
  '--hide-scrollbars',
  '--window-size=1024,640',
  '--force-device-scale-factor=1',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function endpoint() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (response.ok) return (await response.json()).webSocketDebuggerUrl;
    } catch (error) { /* the browser is still starting */ }
    await sleep(250);
  }
  throw new Error('the browser did not expose its debugging endpoint');
}

class Client {
  constructor(socket) {
    this.socket = socket;
    this.next = 1;
    this.pending = new Map();
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      const entry = this.pending.get(message.id);
      if (!entry) return;
      this.pending.delete(message.id);
      if (message.error) entry.reject(new Error(JSON.stringify(message.error)));
      else entry.resolve(message.result);
    });
  }
  send(method, params) {
    const id = this.next;
    this.next += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params: params || {} }));
    });
  }
}

async function main() {
  const webSocketDebuggerUrl = await endpoint();
  const socket = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  const client = new Client(socket);
  const { targetId } = await client.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await client.send('Target.attachToTarget', { targetId, flatten: true });
  const attached = new Client(socket);
  attached.send = (method, params) => {
    const id = attached.next;
    attached.next += 1;
    return new Promise((resolve, reject) => {
      attached.pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params: params || {}, sessionId }));
    });
  };
  await attached.send('Page.enable', {});
  await attached.send('Emulation.setDeviceMetricsOverride', {
    width: 1024, height: 640, deviceScaleFactor: 1, mobile: false,
  });
  let done = 0;
  let broken = 0;
  for (const page of pages) {
    const url = `file:///${path.resolve(page).replace(/\\/g, '/')}`;
    const target = path.join(outputDir, path.basename(page).replace(/\.html$/, '.png'));
    let written = false;
    for (let attempt = 0; attempt < 3 && !written; attempt += 1) {
      await attached.send('Page.navigate', { url });
      await sleep(250);
      const shot = await attached.send('Page.captureScreenshot', {
        format: 'png', captureBeyondViewport: false,
      });
      const buffer = Buffer.from(shot.data, 'base64');
      // A truncated screenshot is a transport condition, so it is captured
      // again rather than accepted.
      if (buffer.length > 512) {
        fs.writeFileSync(target, buffer);
        written = true;
      } else {
        broken += 1;
        await sleep(400);
      }
    }
    if (!written) {
      console.log(`H61_CAPTURE_INCOMPLETE ${path.basename(page)}`);
    }
    done += 1;
    if (done % 50 === 0) console.log(`H61_CAPTURE ${done}/${pages.length}`);
  }
  console.log(`H61_CAPTURE_DONE ${done} incomplete=${broken}`);
  socket.close();
}

main()
  .then(() => { child.kill(); process.exit(0); })
  .catch((error) => { console.error(`H61_CAPTURE_FAILED ${error}`); child.kill(); process.exit(1); });
