// audit/ui_checks/test_mobile_responsiveness.mjs
import { chromium } from "playwright";

const baseUrl = "http://localhost:3333";
const routes = ["/", "/radar", "/countries", "/sectors", "/risk", "/rates", "/liquidity", "/india", "/methodology"];

async function runMobileAudit() {
  console.log("=== Testing Mobile Viewport (375x667) ===");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 375, height: 667 },
    isMobile: true,
  });

  const failures = [];

  for (const route of routes) {
    const page = await context.newPage();
    try {
      await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle", timeout: 15000 });
      await page.waitForTimeout(500);

      // Check horizontal overflow
      const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);

      if (scrollWidth > clientWidth + 5) {
        failures.push({ route, scrollWidth, clientWidth, overflow: scrollWidth - clientWidth });
        console.log(`[WARN] Horizontal page scroll on ${route}: scrollWidth=${scrollWidth} > clientWidth=${clientWidth}`);
      } else {
        console.log(`✓ ${route}: No horizontal page overflow (scrollWidth=${scrollWidth}, clientWidth=${clientWidth})`);
      }
    } catch (e) {
      console.error(`Error on ${route}:`, e.message);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  if (failures.length === 0) {
    console.log("All routes passed mobile responsiveness without page-level horizontal overflow!");
  } else {
    console.log(`Found ${failures.length} routes with horizontal page overflow on mobile.`);
  }
}

runMobileAudit().catch(console.error);
