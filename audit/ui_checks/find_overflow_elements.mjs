// audit/ui_checks/find_overflow_elements.mjs
import { chromium } from "playwright";

async function findOverflow() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 375, height: 667 }, isMobile: true });
  const page = await context.newPage();

  await page.goto("http://localhost:3333/", { waitUntil: "networkidle" });
  await page.waitForTimeout(500);

  const overflowing = await page.evaluate(() => {
    const docWidth = document.documentElement.clientWidth;
    const elements = document.querySelectorAll("*");
    const bad = [];
    elements.forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.right > docWidth + 2) {
        bad.push({
          tag: el.tagName,
          id: el.id,
          className: (el.className || "").toString().slice(0, 100),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
          overflow: Math.round(rect.right - docWidth),
        });
      }
    });
    return bad;
  });

  console.log("Top overflowing elements on /:");
  console.log(overflowing.slice(0, 10));

  await browser.close();
}

findOverflow().catch(console.error);
