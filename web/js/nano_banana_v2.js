// Nano Banana (Vertex AI, node NanoBananaVertexV2): makes the node follow the selected model.
//
// - Each dropdown lists only what the model supports (aspect ratios, resolutions,
//   thinking levels, search modes, locations).
// - Widgets the model does not use are greyed out, not removed.
//
// The rules come from vertex_model_caps.image_ui_caps() in Python, attached to
// the node's `model` input as `vertex_caps`, so this file holds no model data.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "NanoBananaVertexV2";
const FILTERED = ["aspect_ratio", "resolution", "thinking_level", "google_search", "location"];
const DISABLEABLE = ["temperature", "top_p", "seed", "resolution", "thinking_level", "google_search"];

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function setDisabled(w, disabled) {
  if (!w) return;
  w.disabled = disabled;
  // The seed's "control after generate" companion follows the seed.
  for (const linked of w.linkedWidgets ?? []) linked.disabled = disabled;
}

function applyModel(node, allCaps) {
  const modelWidget = widget(node, "model");
  const caps = modelWidget && allCaps[modelWidget.value];
  if (!caps) return;

  for (const name of FILTERED) {
    const w = widget(node, name);
    const options = caps.options[name];
    if (!w || !options) continue; // no list = the widget is disabled below; keep its value
    w.options.values = options;
    if (!options.includes(w.value)) {
      // Keep "auto"/"off" if the model has it, else fall back to the first choice.
      w.value = options.includes("auto") ? "auto" : options.includes("off") ? "off" : options[0];
      w.callback?.(w.value);
    }
  }

  for (const name of DISABLEABLE) setDisabled(widget(node, name), caps.disabled.includes(name));

  node.updateComputedDisabled?.();
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "VertexAPI.NanoBananaV2",

  // A batch where some generations failed still returns the rest; say so on screen.
  setup() {
    api.addEventListener("vertexapi.partial_failure", ({ detail }) => {
      app.extensionManager?.toast?.add({
        severity: "warn",
        summary: detail.summary,
        detail: detail.detail,
        life: 15000,
      });
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;
    const allCaps = nodeData.input?.required?.model?.[1]?.vertex_caps;
    if (!allCaps) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const modelWidget = widget(this, "model");
      if (modelWidget) {
        const callback = modelWidget.callback;
        const node = this;
        modelWidget.callback = function () {
          const r = callback?.apply(this, arguments);
          applyModel(node, allCaps);
          return r;
        };
      }
      applyModel(this, allCaps);
      return result;
    };

    // Saved workflows restore widget values after creation; re-apply on load.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      applyModel(this, allCaps);
      return result;
    };
  },
});
