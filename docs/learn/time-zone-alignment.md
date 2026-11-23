# Learning Note: Time-Zone Alignment Across International Markets

**Topic:** Non-Synchronous Closes and Attribution Synchronization  
**Reading Time:** 3 minutes (~500 words)

---

## 1. Intuition in Plain Language

Imagine you are analyzing the National Stock Exchange of India (Nifty 50) and the US 10-Year Treasury Yield.
- Nifty 50 closes in Mumbai at **15:30 IST**.
- At that exact moment, New York time is **05:00 AM ET**. The US bond market hasn't even opened for regular trading!
- The US 10-Year Treasury closes at **15:00 ET**, which corresponds to **01:30 AM IST the next day**.

If an algorithm naively matches same-date rows (e.g., `2024-10-15` Nifty close with `2024-10-15` US 10Y close), the US bond move that happened at 2:00 PM ET occurred **9 hours AFTER the Indian market already closed**.

The Indian market could not possibly have reacted to a US yield move that hadn't happened yet! Instead, the Indian market will react to that US bond move the **following morning** (`2024-10-16`). Naive same-date regression creates spurious, lagged coefficients and ruins attribution.

---

## 2. The Timestamp Alignment Rule

For target market $T$ with local closing timestamp $C_T(d)$ on date $d$:
Each global driver factor $f$ must be sampled as of its **last available close at or before** the target's close:

$$x_f(d) = \ln\left(\frac{P_f(\text{last close} \le C_T(d))}{P_f(\text{last close} \le C_T(d-1))}\right)$$

```text
Target: Tokyo (N225) Closes 15:30 JST (06:30 UTC)
US Factor (S&P 500) Closes 16:00 ET (21:00 UTC)

Timeline on Date d:
06:30 UTC: Tokyo Closes for Day d
           --> S&P 500 for Day d has NOT happened yet!
           --> Last available S&P 500 close is Day d-1 (yesterday).
           --> Correct aligned factor move for Tokyo(d) is S&P 500 from (d-2) to (d-1)!
```

---

## 3. Why Naive Same-Date Attribution Fails

If an Asian market responds with a 1-day lag to overnight US technology stock earnings, naive same-date OLS regression reports:
- Low beta to US Equities ($R^2 \approx 0$).
- Huge unexplained residual.

Under time-zone timestamp alignment:
- The previous US session is correctly recognized as the causal information set available to Asian traders at their open.
- The true beta is recovered accurately.

---

## 4. The 5-Day Default View

While 1-day returns are acutely sensitive to minute-by-minute closing differences, **5-day and 1-month returns** naturally attenuate the synchronization gap because a 5-day return spans 120 hours of trading, of which the asynchronous gap is only a tiny fraction (~8 hours).

Therefore, Global Market Radar defaults the Move Decomposer to **5-day horizon**, and tags 1-day decompositions for Asian/Pacific markets with a visible **"Synchronization-Sensitive"** notice.

---

## 5. Where It Lives in the Code

- Configuration: [pipeline/config/calendars.yaml](file:///d:/MacroMonitor/pipeline/config/calendars.yaml).
- Specification: Section 3.4 of `docs/MASTER_PLAN.md` and [ADR 013](file:///d:/MacroMonitor/docs/decisions/0013-move-decomposer-and-time-alignment.md).
