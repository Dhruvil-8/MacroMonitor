// audit/ui_checks/run_phase_e_playwright.mjs
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const baseUrl = "http://localhost:3333";
const resultsDir = path.resolve(__dirname, "../results");
fs.mkdirSync(resultsDir, { recursive: true });

console.log("=== Phase E: Frontend Data-to-Pixel Audit ===");

const routesToTest = [
  "/",
  "/radar",
  "/countries",
  "/countries/us",
  "/countries/in",
  "/sectors",
  "/drivers",
  "/risk",
  "/rates",
  "/positioning",
  "/events",
  "/connectedness",
  "/fundamentals",
  "/liquidity",
  "/india",
  "/context/spx",
  "/context/nifty50",
  "/briefing",
  "/track-record",
  "/data-health",
  "/methodology",
];

const auditResults = [];
const pageSnapshots = [];
const findings = [];

async function runAudit() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });

  for (const route of routesToTest) {
    console.log(`\nAuditing route: ${route}`);
    const page = await context.newPage();
    const consoleErrors = [];
    const failedRequests = [];
    const requests = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    page.on("requestfailed", (req) => {
      failedRequests.push(`${req.method()} ${req.url()}: ${req.failure()?.errorText}`);
    });

    page.on("response", (res) => {
      if (res.status() >= 400) {
        failedRequests.push(`HTTP ${res.status()} on ${res.url()}`);
      }
      if (res.url().includes("/data/v1/")) {
        requests.push(res.url());
      }
    });

    try {
      await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle", timeout: 15000 });
      // Allow ECharts rendering animation to settle
      await page.waitForTimeout(1000);

      // 1. Extract StatBlocks
      const stats = await page.$$eval('[data-testid="stat-block"], .stat-block', (els) => {
        return els.map((el) => ({
          label: el.querySelector(".stat-label, label")?.textContent?.trim() || "",
          value: el.querySelector(".stat-value, .value, [data-stat-value]")?.textContent?.trim() || "",
          unit: el.querySelector(".stat-unit, .unit")?.textContent?.trim() || "",
        }));
      });

      // 2. Extract ECharts instances via DOM evaluation
      const echartsInfo = await page.evaluate(() => {
        const chartEls = document.querySelectorAll("div[_echarts_instance_], [data-echarts]");
        const out = [];
        chartEls.forEach((el, idx) => {
          try {
            // @ts-ignore
            const inst = window.echarts?.getInstanceByDom(el);
            if (inst) {
              const opt = inst.getOption();
              const seriesData = (opt.series || []).map((s) => ({
                name: s.name,
                type: s.type,
                dataLength: (s.data || []).length,
                samplePoints: (s.data || []).slice(0, 5),
              }));
              out.push({
                index: idx,
                seriesCount: (opt.series || []).length,
                series: seriesData,
                title: opt.title?.[0]?.text || "",
              });
            } else {
              out.push({ index: idx, status: "instance_not_found" });
            }
          } catch (e) {
            out.push({ index: idx, error: String(e) });
          }
        });
        return out;
      });

      // 3. Extract Headings and Cards
      const headings = await page.$$eval("h1, h2, h3", (els) => els.map((e) => e.textContent?.trim() || ""));
      const cardTitles = await page.$$eval('[data-card-title], .card-title, h3', (els) => els.map((e) => e.textContent?.trim() || ""));

      // 4. Test AuditDrawer trigger if present
      const auditTrigger = await page.$('button[aria-label*="audit" i], button[aria-label*="method" i], [data-audit-trigger]');
      let auditDrawerWorking = false;
      if (auditTrigger) {
        try {
          await auditTrigger.click();
          await page.waitForTimeout(300);
          const drawerVisible = await page.$eval('[role="dialog"], .audit-drawer', (el) => el && el.clientHeight > 0).catch(() => false);
          auditDrawerWorking = drawerVisible;
          // Close drawer
          await page.keyboard.press("Escape");
        } catch (_) {}
      }

      const snapshot = {
        route,
        headings: headings.slice(0, 5),
        statsCount: stats.length,
        statsSample: stats.slice(0, 3),
        echartsCount: echartsInfo.length,
        echarts: echartsInfo,
        requestsCount: requests.length,
        consoleErrorsCount: consoleErrors.length,
        failedRequestsCount: failedRequests.length,
        auditDrawerWorking,
      };
      pageSnapshots.push(snapshot);

      const hasCriticalError = consoleErrors.length > 0 || failedRequests.length > 0;
      auditResults.push({
        route,
        status: hasCriticalError ? "FAIL" : "PASS",
        stats_found: stats.length,
        charts_found: echartsInfo.length,
        requests_count: requests.length,
        console_errors: consoleErrors.join(" | "),
        failed_requests: failedRequests.join(" | "),
      });

      console.log(`✓ ${route}: Charts=${echartsInfo.length}, Stats=${stats.length}, Requests=${requests.length}, Errors=${consoleErrors.length}`);
    } catch (err) {
      console.error(`✗ Error on ${route}:`, err.message);
      auditResults.push({
        route,
        status: "ERROR",
        stats_found: 0,
        charts_found: 0,
        requests_count: requests.length,
        console_errors: err.message,
        failed_requests: failedRequests.join(" | "),
      });
    } finally {
      await page.close();
    }
  }

  await browser.close();

  // Write outputs
  fs.writeFileSync(path.join(resultsDir, "ui_page_snapshots.json"), JSON.stringify(pageSnapshots, null, 2));

  const csvHeader = "route,status,stats_found,charts_found,requests_count,console_errors,failed_requests\n";
  const csvRows = auditResults.map((r) =>
    `"${r.route}","${r.status}",${r.stats_found},${r.charts_found},${r.requests_count},"${(r.console_errors || "").replace(/"/g, '""')}","${(r.failed_requests || "").replace(/"/g, '""')}"`
  ).join("\n");

  fs.writeFileSync(path.join(resultsDir, "ui_value_equality_results.csv"), csvHeader + csvRows);
  console.log(`\nUI Audit complete. Processed ${routesToTest.length} routes.`);
  console.log(`Snapshots saved to: ${path.join(resultsDir, "ui_page_snapshots.json")}`);
  console.log(`Results saved to: ${path.join(resultsDir, "ui_value_equality_results.csv")}`);
}

runAudit().catch(console.error);
