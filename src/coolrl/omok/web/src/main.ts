import { mount } from "svelte";
import { logDebug, logError, logInfo } from "./util/logger";

import App from "./App.svelte";
import "./styles.css";

logInfo("OmokMain", "boot", {
  isProductionMode: import.meta.env?.MODE === "production",
});

const target = document.getElementById("app");

logDebug("OmokMain", "targetResolved", {
  exists: !!target,
});

if (!target) {
  logError("OmokMain", "missingTarget", { targetId: "app" });
  throw new Error("Missing #app mount target");
}

const result = mount(App, { target });
logInfo("OmokMain", "mounted");

export default result;
