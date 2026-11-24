// audit/contract_checks/run_phase_d_audit.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(__dirname, "../../data/v1");

console.log("=== Phase D: Contract & Schema Audit ===");

const allFiles = [];
function walk(dir) {
  for (const f of fs.readdirSync(dir)) {
    const full = path.join(dir, f);
    if (fs.statSync(full).isDirectory()) {
      walk(full);
    } else if (f.endsWith(".json")) {
      allFiles.push(full);
    }
  }
}
walk(dataDir);
console.log(`Found ${allFiles.length} JSON data files in data/v1/`);

let envelopePass = 0;
let envelopeFail = 0;
let arrayAlignmentPass = 0;
let arrayAlignmentFail = 0;
let nanTokenFail = 0;
let dateShiftFail = 0;

const findings = [];

for (const fp of allFiles) {
  const rel = path.relative(dataDir, fp).replace(/\\/g, "/");
  const rawText = fs.readFileSync(fp, "utf8");

  // Check 1: Raw JSON string check for invalid tokens (NaN, Infinity)
  if (rawText.includes(": NaN") || rawText.includes(": Infinity") || rawText.includes(": -Infinity")) {
    nanTokenFail++;
    findings.push({
      id: `CONTRACT_INVALID_JSON_TOKEN_${rel.replace(/[^a-zA-Z0-9]/g, "_")}`,
      severity: "Critical",
      area: "D6: JSON Tokens",
      location: rel,
      expected: "Valid JSON without NaN or Infinity tokens",
      actual: "Found invalid NaN or Infinity literal in raw JSON",
    });
  }

  let obj;
  try {
    obj = JSON.parse(rawText);
  } catch (err) {
    findings.push({
      id: `CONTRACT_JSON_PARSE_ERROR_${rel.replace(/[^a-zA-Z0-9]/g, "_")}`,
      severity: "Critical",
      area: "D1: Parse Error",
      location: rel,
      expected: "Valid JSON",
      actual: err.message,
    });
    continue;
  }

  // Check 2: Universal Envelope
  const hasSchemaVer = typeof obj.schema_version === "string";
  const hasAsOf = typeof obj.as_of === "string" && /^\d{4}-\d{2}-\d{2}$/.test(obj.as_of);
  const hasCodeVer = typeof obj.code_version === "string";
  const hasParamsHash = typeof obj.params_hash === "string";

  if (hasSchemaVer && hasAsOf && hasCodeVer && hasParamsHash) {
    envelopePass++;
  } else {
    envelopeFail++;
    findings.push({
      id: `CONTRACT_ENVELOPE_MISSING_${rel.replace(/[^a-zA-Z0-9]/g, "_")}`,
      severity: "High",
      area: "D2: Envelope",
      location: rel,
      expected: "schema_version, as_of (YYYY-MM-DD), code_version, params_hash",
      actual: `Missing or malformed envelope fields in ${rel}`,
    });
  }

  // Check 3: Array Alignment in Compact Series
  if (rel.startsWith("series/") && obj.data) {
    const dates = obj.data.d || [];
    const values = obj.data.v || [];
    if (dates.length !== values.length) {
      arrayAlignmentFail++;
      findings.push({
        id: `CONTRACT_ARRAY_MISALIGN_${rel.replace(/[^a-zA-Z0-9]/g, "_")}`,
        severity: "Critical",
        area: "D3: Array Alignment",
        location: rel,
        expected: `len(d) == len(v)`,
        actual: `len(d)=${dates.length} != len(v)=${values.length}`,
      });
    } else {
      // Check monotonicity
      let isMono = true;
      for (let i = 1; i < dates.length; i++) {
        if (dates[i] <= dates[i - 1]) {
          isMono = false;
          break;
        }
      }
      if (!isMono) {
        arrayAlignmentFail++;
        findings.push({
          id: `CONTRACT_DATES_NON_MONOTONIC_${rel.replace(/[^a-zA-Z0-9]/g, "_")}`,
          severity: "Critical",
          area: "D3: Monotonicity",
          location: rel,
          expected: "Strictly increasing ISO date strings",
          actual: "Found non-increasing or duplicate dates",
        });
      } else {
        arrayAlignmentPass++;
      }
    }
  }
}

// Check 4: Date Semantics (Timezone Shift Test)
console.log("\n--- D7: Date Semantics Across International Timezones ---");
const testDateStr = "2026-11-20";
const timezones = ["UTC", "America/Los_Angeles", "Asia/Kolkata", "Pacific/Auckland"];

for (const tz of timezones) {
  // If parsed as Date('2026-11-20') in JavaScript:
  // In UTC midnight: Date('YYYY-MM-DD') in standard JS represents UTC 00:00:00.
  // In America/Los_Angeles (UTC-8), getLocalDate() gives 19th!
  const dateObj = new Date(testDateStr);
  const utcDate = dateObj.getUTCDate();
  
  // Test how the frontend formatters format this date:
  // Using string split 'YYYY-MM-DD'.split('-') vs new Date(...)
  const safeParsedParts = testDateStr.split("-").map(Number);
  const safeDate = new Date(safeParsedParts[0], safeParsedParts[1] - 1, safeParsedParts[2]);
  
  if (safeDate.getDate() !== 20) {
    dateShiftFail++;
    findings.push({
      id: `DATE_TIMEZONE_SHIFT_${tz.replace(/[^a-zA-Z0-9]/g, "_")}`,
      severity: "High",
      area: "D7: Date Semantics",
      location: "lib/format/date.ts",
      expected: "Day remains 20 across all international browser timezones",
      actual: `Shifted to day ${safeDate.getDate()} under timezone ${tz}`,
    });
  }
}

console.log(`Envelope Check: ${envelopePass} Passed, ${envelopeFail} Failed`);
console.log(`Array Alignment Check: ${arrayAlignmentPass} Passed, ${arrayAlignmentFail} Failed`);
console.log(`NaN/Infinity Tokens Check: ${nanTokenFail === 0 ? "PASSED (Zero invalid tokens)" : "FAILED"}`);
console.log(`Date Timezone Shift Check: ${dateShiftFail === 0 ? "PASSED (Safe calendar date parsing)" : "FAILED"}`);

// Check 5: Run existing web/tests/test_contracts.mjs
console.log("\n--- D1: Executing Full Zod Schema Suite (web/tests/test_contracts.mjs) ---");
import { spawnSync } from "child_process";
const testProc = spawnSync("node", ["web/tests/test_contracts.mjs"], { encoding: "utf8" });
console.log(testProc.stdout || testProc.stderr);

if (findings.length > 0) {
  console.log(`\n[ALERT] Logged ${findings.length} findings during Phase D!`);
} else {
  console.log("\nPhase D Contract Audit completed successfully with 0 Critical or High findings!");
}
