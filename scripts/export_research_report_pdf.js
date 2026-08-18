const path = require("path");
const puppeteer = require("/tmp/node_modules/puppeteer");

const htmlPath = path.resolve(__dirname, "../docs/html/research-report.html");
const outPath = path.resolve(__dirname, "../docs/research-report-slides.pdf");

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 810, deviceScaleFactor: 2 });
  await page.emulateMediaType("print");
  await page.goto("file://" + htmlPath, { waitUntil: "networkidle0", timeout: 30000 });
  await page.evaluate(() => {
    document.documentElement.setAttribute("data-theme", "light");
    document.querySelectorAll(".slide").forEach((s) => s.classList.add("in"));
    document.querySelectorAll(".anim").forEach((el) => {
      el.style.opacity = "1";
      el.style.transform = "none";
    });
  });
  await new Promise((r) => setTimeout(r, 800));
  await page.pdf({
    path: outPath,
    width: "13.333in",
    height: "7.5in",
    printBackground: true,
    preferCSSPageSize: false,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
  });
  await browser.close();
  console.log("OK:", outPath);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
