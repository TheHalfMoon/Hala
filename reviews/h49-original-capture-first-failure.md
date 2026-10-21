# H49 first Edge capture failure (2026-10-21)

Original first run generated exactly 96 fictional HTML scenes and attempted 96 original browser screenshot/click replays. It stopped on `h49-traffic_services-01`, **before completing** the capture or creating a trusted capture.json. Original terminal error:

```text
{"created": 96}
Error: True postcondition mismatch on golden target h49-traffic_services-01
    at tools/h49_offline_capture.js:68:14
```

Independent reproducer on the original untouched HTML confirmed `justice_execution-00` passed with expected selected=0, clicks=1. On `traffic_services-01` the intended sixth button (index=5, key=0) existed in the DOM, but headless browser `page.locator(...).click()` produced **no** body selection state (`after={}`). This was an out-of-viewport vertical layout: the sidebar/table row spacing pushed the bottom button beyond the 640px viewport. An unqualified auto-generated synthetic DOM must never be treated as a valid visible button.

Repair changes H49 original-source CSS table/side rows from 75px to 62px, gap from 9px to 7px, and adds strict browser-captured geometry checks that every control is fully within the 1024×610 unobscured viewport region before writing screenshots. Only this new H49 worktree's partial uncommitted generated artifacts are regenerated; the original failed outcome is preserved here. No government or clinical website was accessed, no model training occurred.
