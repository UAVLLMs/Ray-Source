/*
 * Frontend controller for the Manual Chat demo.
 *
 * This file is intentionally dependency-free: no React/Vue build step, no
 * package install, and no bundler. The page is served directly by `server.py`,
 * which matters for contest rehearsal because teammates can open the demo on a
 * local machine or through a public tunnel without a frontend build pipeline.
 *
 * The comments below are deliberately detailed because this demo is being used
 * by multiple people at the same time. Future edits should preserve the split
 * between customer-service mode and manual mode unless the team explicitly
 * changes the backend routing contract.
 */

// Browser requests stay on the web-client origin. The web server owns the
// retrieval-service token, so no backend credential is shipped to the browser.
const CHAT_API_TOKEN = "";

// Official REST endpoint implemented by `server.py`.
const CHAT_ENDPOINT = "/raysource-api/chat";
// `/chat` is the gateway's established transactional-service route (8011).
// The lightweight 8014 route deliberately handles manuals only.
const CUSTOMER_SERVICE_CHAT_ENDPOINT = "/chat";
const PROGRESS_ENDPOINT = "/raysource-api/progress";
const MODEL_PROFILE_ENDPOINT = "/raysource-api/model-profile";
const MODEL_PROFILE_SWITCH_ENDPOINT = "/raysource-api/model-profile/switch";
// Citation links use the locator page. It retains the navigation script that
// expands the relevant manual and applies excerpt/image highlights.
const MANUAL_INDEX_ENDPOINT = "/rag/manual-locate/";
const IMAGE_CAPTION_ENDPOINT = "/rag/image-caption";
const TRANSLATE_ENDPOINT = "/raysource-api/translate";
const ACCOUNT_ME_ENDPOINT = "/raysource-api/account/me";
const ACCOUNT_REGISTER_ENDPOINT = "/raysource-api/account/register";
const ACCOUNT_LOGIN_ENDPOINT = "/raysource-api/account/login";
const ACCOUNT_LOGOUT_ENDPOINT = "/raysource-api/account/logout";
const CONVERSATIONS_ENDPOINT = "/raysource-api/account/conversations";
const FEEDBACK_ENDPOINT = "/raysource-api/feedback";
const imageCaptionCache = new Map();

// Timeout values mirror the backend budget. Customer-template imitation can
// need a full model round plus a repair round, so the browser must not abort
// before `server.py` has a chance to return `answer_mode`.
const TEXT_TIMEOUT_MS = 120_000;
const MULTIMODAL_TIMEOUT_MS = 180_000;
const MAX_HISTORY_TURNS = 6;
const MAX_HISTORY_CONTEXT_CHARS = 1200;
const CONTEXT_PACKET_VERSION = 1;
const HISTORY_ONLY_RE = /(?:只|仅).{0,12}(?:根据|使用).{0,12}(?:上一轮|上轮|历史|刚才|前面).{0,20}(?:回答|复述|说明)|(?:不要|无需|不用).{0,12}(?:重新)?(?:检索|搜索|查(?:询)?手册)/i;
// Manual answers render immediately. Customer-service answers intentionally keep
// a short, human-readable streaming cadence so instant template/cache replies do
// not look like an unverified canned response.
const MIN_PROGRESS_MS = 0;
const MIN_MULTIMODAL_PROGRESS_MS = 0;
const CUSTOMER_STREAM_MIN_MS = 12_000;
const CUSTOMER_STREAM_TARGET_MS = 14_000;
const CUSTOMER_STREAM_MAX_MS = 17_000;
const MANUAL_STREAM_MIN_MS = 4_000;
const MANUAL_STREAM_TARGET_MS = 5_500;
const MANUAL_STREAM_MAX_MS = 8_000;
const REMOTE_MEDIA_URL_PATTERN = /\bhttps?:\/\/[^\s<>"']+/i;
const MEMORY_EPOCH_STORAGE_KEY = "raysource_product_memory_epochs";
const PRODUCT_MEMORY_STORAGE_KEY = "raysource_product_memories_v1";
const MOBILE_LOGBAR_WIDTH_STORAGE_KEY = "raysource_mobile_logbar_width_v1";

let mobileLogbarResizeState = null;
let mobileLogbarSwipeState = null;

function loadProductMemoryEpochs() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(MEMORY_EPOCH_STORAGE_KEY) || "{}");
    return new Map(Object.entries(parsed).filter(([product, epoch]) => product && typeof epoch === "string"));
  } catch {
    return new Map();
  }
}

function persistProductMemoryEpochs() {
  window.localStorage.setItem(
    MEMORY_EPOCH_STORAGE_KEY,
    JSON.stringify(Object.fromEntries(state.productMemoryEpochs)),
  );
}

function loadProductMemories() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(PRODUCT_MEMORY_STORAGE_KEY) || "{}");
    return new Map(Object.entries(parsed).filter(([product, turns]) => (
      product && Array.isArray(turns)
    )).map(([product, turns]) => [product, turns.slice(-MAX_HISTORY_TURNS)]));
  } catch {
    return new Map();
  }
}

function persistProductMemories() {
  window.localStorage.setItem(
    PRODUCT_MEMORY_STORAGE_KEY,
    JSON.stringify(Object.fromEntries(state.productMemories)),
  );
}

function hasRemoteMediaUrl(question) {
  // The retrieval service safely inspects every user-provided HTTP(S) URL. A
  // direct image is downloaded as-is; an HTML page may contribute its OpenGraph
  // representative image. Treat either case as multimodal in the browser so it
  // receives the same progress and timeout budget as a local image upload.
  return REMOTE_MEDIA_URL_PATTERN.test(String(question || ""));
}

// The backend also knows this product name; the browser sends it only as a hint.
// The final customer/manual split still comes from `/chat.answer_mode`.
const CUSTOMER_SERVICE_PRODUCT = "客服售后";

// Transaction / after-sales facts identify a request to the seller or
// marketplace, not a manual question.  Product words, the active menu item,
// and previous turns are intentionally excluded from this decision.
// Keep this high-precision: product operations such as "更换电池" must remain
// on the manual route, whereas "申请换货" and "订单退款" must not be guessed as
// a random product manual.
const CUSTOMER_SERVICE_INTENT_RE = /(?:订单|快递|物流|发货|收货|签收|退货|退款|换货|退换|售后|投诉|发票|运费|运费险|补发|少发|漏发|错发|商品描述不符|质量问题|保质期|生产日期|以旧换新|智能客服|联系客服|客服(?:人员)?|优惠券|试用(?:装|期)?|上门(?:安装|检修)|额外(?:收费|收取|配件费)|尺寸(?:差价|不合适)|更换(?:成)?(?:更大|更小)?尺寸|申请(?:退货|退款|换货|售后|维修)|维修(?:费用|报价|进度|服务)|(?:质保|保修)期|终身维修|纸质版说明书|电子版(?:说明书)?在哪里|商家|店铺|卖家|订单号|快递单号|\b(?:my\s+order|order\s+(?:number|status|tracking)|shipping|delivery|tracking|refund|return|exchange|after[- ]?sales|invoice|complaint|customer\s*service|seller|merchant|missing\s*item|replacement\s*order)\b)/i;

function isCustomerServiceQuestion(question) {
  return CUSTOMER_SERVICE_INTENT_RE.test(String(question || ""));
}

// Single source of truth for transient UI state. Keeping state in one object
// makes it easier to reason about concurrency: only one submit can be active
// while `busy` is true, and image attachment/session data are always read from
// here before a request is sent.
const state = {
  data: null,
  activeProduct: null,
  activeItem: null,
  attachment: null,
  busy: false,
  questionMenuCloseTimer: null,
  sessionId: window.localStorage.getItem("ragv6_session_id") || null,
  // Memory is opt-in and isolated by canonical product name. A -> B -> A
  // therefore restores A's bucket without leaking A into B.
  productMemories: loadProductMemories(),
  productMemoryEpochs: loadProductMemoryEpochs(),
  // Per-question RAG process records. Each chat turn owns one record; the right
  // sidebar is a viewer that renders whichever record is currently active, so the
  // log shows ONE question's flow at a time (ChatGPT-style), not a running pile.
  processes: new Map(),
  activeProcessId: null,
  live: null,
  procSeq: 0,
  modelProfiles: [],
  activeModelProfile: null,
  modelMenuOpen: false,
  reasoningEffort: "medium",
  reasoningMenuOpen: false,
  authUser: null,
  authMode: "guest",
  authFormMode: "login",
  conversations: [],
  activeConversationId: null,
};

// Cached DOM references. Querying once avoids repeatedly walking the DOM during
// progress animation and message rendering.
const els = {
  productList: document.querySelector("#productList"),
  productSearch: document.querySelector("#productSearch"),
  currentConversation: document.querySelector("#currentConversation"),
  currentConversationTitle: document.querySelector("#currentConversationTitle"),
  conversationList: document.querySelector("#conversationList"),
  conversationNote: document.querySelector("#conversationNote"),
  contextModeBadge: document.querySelector("#contextModeBadge"),
  activeProduct: document.querySelector("#activeProduct"),
  manualIndexBtn: document.querySelector("#manualIndexBtn"),
  messages: document.querySelector("#messages"),
  composer: document.querySelector("#composer"),
  attachmentPreview: document.querySelector("#attachmentPreview"),
  imageInput: document.querySelector("#imageInput"),
  uploadBtn: document.querySelector("#uploadBtn"),
  questionInput: document.querySelector("#questionInput"),
  questionMenuBtn: document.querySelector("#questionMenuBtn"),
  questionMenu: document.querySelector("#questionMenu"),
  modelMenuBtn: document.querySelector("#modelMenuBtn"),
  modelMenu: document.querySelector("#modelMenu"),
  activeModelLabel: document.querySelector("#activeModelLabel"),
  activeModelIcon: document.querySelector("#activeModelIcon"),
  reasoningSwitcher: document.querySelector("#reasoningSwitcher"),
  reasoningMenuBtn: document.querySelector("#reasoningMenuBtn"),
  reasoningMenu: document.querySelector("#reasoningMenu"),
  activeReasoningLabel: document.querySelector("#activeReasoningLabel"),
  questionField: document.querySelector(".question-field"),
  historyContextToggle: document.querySelector("#historyContextToggle"),
  historyContextScope: document.querySelector("#historyContextScope"),
  historyContextCount: document.querySelector("#historyContextCount"),
  clearHistoryContext: document.querySelector("#clearHistoryContext"),
  sendBtn: document.querySelector("#sendBtn"),
  progressPanel: document.querySelector("#progressPanel"),
  progressStatusCard: document.querySelector(".progress-status-card"),
  progressBar: document.querySelector("#progressBar"),
  progressStage: document.querySelector("#progressStage"),
  progressStatus: document.querySelector("#progressStatus"),
  progressElapsed: document.querySelector("#progressElapsed"),
  progressPercent: document.querySelector("#progressPercent"),
  progressLog: document.querySelector("#progressLog"),
  progressSummary: document.querySelector("#progressSummary"),
  progressImages: document.querySelector("#progressImages"),
  progressThumbs: document.querySelector("#progressThumbs"),
  logbar: document.querySelector("#logbar"),
  newChatBtn: document.querySelector("#newChatBtn"),
  logbarToggle: document.querySelector("#logbarToggle"),
  mobileSidebarToggle: document.querySelector("#mobileSidebarToggle"),
  mobileLogbarToggle: document.querySelector("#mobileLogbarToggle"),
  mobileLogbarClose: document.querySelector("#mobileLogbarClose"),
  mobileLogbarExpand: document.querySelector("#mobileLogbarExpand"),
  mobileLogbarResize: document.querySelector("#mobileLogbarResize"),
  mobileManualIndexBtn: document.querySelector("#mobileManualIndexBtn"),
  mobileDrawerBackdrop: document.querySelector("#mobileDrawerBackdrop"),
  appShell: document.querySelector(".app-shell"),
  accountButton: document.querySelector("#accountButton"),
  accountAvatar: document.querySelector("#accountAvatar"),
  accountName: document.querySelector("#accountName"),
  accountHint: document.querySelector("#accountHint"),
  accountMenu: document.querySelector("#accountMenu"),
  accountMenuIdentity: document.querySelector("#accountMenuIdentity"),
  accountLoginAction: document.querySelector("#accountLoginAction"),
  accountLogoutAction: document.querySelector("#accountLogoutAction"),
  authModal: document.querySelector("#authModal"),
  authClose: document.querySelector("#authClose"),
  loginTab: document.querySelector("#loginTab"),
  registerTab: document.querySelector("#registerTab"),
  authForm: document.querySelector("#authForm"),
  authUsername: document.querySelector("#authUsername"),
  authPassword: document.querySelector("#authPassword"),
  authError: document.querySelector("#authError"),
  authSubmit: document.querySelector("#authSubmit"),
  guestContinue: document.querySelector("#guestContinue"),
};

// Product -> icon key map. The icons are hand-written SVG symbols below, not
// manual illustrations. This matches the current UI requirement: sidebar icons
// should be clean product symbols, while answer images still come from manuals.
const ICON_BY_PRODUCT = {
  客服售后: "support",
  洗碗机: "dishwasher",
  空调: "airConditioner",
  空气净化器: "airPurifier",
  电钻: "drill",
  健身追踪器: "fitnessTracker",
  健身单车: "exerciseBike",
  吹风机: "blower",
  相机: "camera",
  混合即时相机: "instantCamera",
  摩托艇: "jetSki",
  水上摩托: "jetSki",
  烤架: "grill",
  咖啡机: "coffee",
  耳机: "headphones",
  电子阅读器: "ereader",
  传真机: "fax",
  空气炸锅: "airFryer",
  电动牙刷: "toothbrush",
  蒸汽清洁机: "steamCleaner",
  人体工学椅: "chair",
  VR头显: "vr",
  "热泵/处理器单元": "heatPump",
  固定电话: "phone",
  割草机: "mower",
  微波炉: "microwave",
  主板: "motherboard",
  扫地机器人: "robotVacuum",
  雪地摩托: "snowmobile",
  "电视/天线": "tv",
  "对讲机/通信设备": "walkieTalkie",
  冰箱: "fridge",
  烤箱: "oven",
  可编程温控器: "thermostat",
  蓝牙鼠标: "mouse",
  水泵: "waterPump",
  儿童电动摩托车: "kidsBike",
  发电机: "generator",
  功能键盘: "keyboard",
  其他产品: "product",
};

// Inline SVG icon library. Each value is generated by `iconSvg`, so every icon
// shares the same 32x32 viewBox and stroke styling from CSS. The large literal
// object is intentionally kept in this file because there is no build step and
// the icons must work offline.
const SVG_ICONS = {
  support: iconSvg('<path d="M9 19a7 7 0 0 1 14 0"/><path d="M9 19v3a3 3 0 0 0 3 3h2"/><path d="M23 19v2a4 4 0 0 1-4 4h-3"/><rect x="6" y="16" width="4" height="7" rx="2"/><rect x="22" y="16" width="4" height="7" rx="2"/><path d="M13 10h6"/>'),
  dishwasher: iconSvg('<rect x="7" y="5" width="18" height="22" rx="3"/><path d="M7 11h18"/><path d="M11 17h10"/><path d="M11 21h8"/><path d="M19 8h2"/><path d="M13 14c-2 2-2 4 0 6"/><path d="M17 14c-2 2-2 4 0 6"/>'),
  airConditioner: iconSvg('<rect x="5" y="8" width="22" height="11" rx="3"/><path d="M9 15h14"/><path d="M12 22c1.4-1 2.8-1 4.2 0"/><path d="M18 25c1.4-1 2.8-1 4.2 0"/><circle cx="24" cy="12" r="1"/>'),
  airPurifier: iconSvg('<rect x="10" y="5" width="12" height="22" rx="3"/><path d="M13 11h6"/><path d="M13 15h6"/><path d="M13 19h6"/><path d="M6 10c-2 2-2 5 0 7"/><path d="M26 10c2 2 2 5 0 7"/>'),
  drill: iconSvg('<path d="M6 13h12l5 3-5 3H6z"/><path d="M12 19v7h5l2-7"/><path d="M23 16h4"/><path d="M9 13V9h8v4"/><path d="M14 22h3"/>'),
  fitnessTracker: iconSvg('<rect x="11" y="10" width="10" height="12" rx="3"/><path d="M13 10l1-5h4l1 5"/><path d="M13 22l1 5h4l1-5"/><path d="M14 16h4"/>'),
  exerciseBike: iconSvg('<circle cx="9" cy="22" r="4"/><circle cx="23" cy="22" r="4"/><path d="M9 22l6-10 4 10"/><path d="M15 12h5"/><path d="M16 12l-4 10"/><path d="M20 10l3 3"/><path d="M14 8h4"/>'),
  blower: iconSvg('<path d="M7 12h10l6 4-6 4H7z"/><path d="M12 20l-2 7h5l3-7"/><path d="M23 16h4"/><path d="M9 16h3"/>'),
  camera: iconSvg('<rect x="6" y="10" width="20" height="14" rx="3"/><path d="M11 10l2-3h6l2 3"/><circle cx="16" cy="17" r="4"/><path d="M23 13h1"/>'),
  instantCamera: iconSvg('<rect x="7" y="6" width="18" height="16" rx="3"/><circle cx="16" cy="14" r="4"/><path d="M11 22v5h10v-5"/><path d="M11 10h3"/><path d="M21 10h1"/>'),
  jetSki: iconSvg('<path d="M6 20c5 2 14 2 20-1"/><path d="M9 18l4-5h7l4 5"/><path d="M14 13l-1-4h5l2 4"/><path d="M5 25c3-2 5-2 8 0 3 2 5 2 8 0 2-1.3 4-1.7 6-.6"/>'),
  grill: iconSvg('<rect x="7" y="11" width="18" height="9" rx="2"/><path d="M9 11c0-3 4-3 4-6"/><path d="M15 11c0-3 4-3 4-6"/><path d="M10 20l-3 6"/><path d="M22 20l3 6"/><path d="M10 15h12"/><path d="M13 11v9M17 11v9M21 11v9"/>'),
  coffee: iconSvg('<path d="M10 13h12v6a6 6 0 0 1-6 6 6 6 0 0 1-6-6z"/><path d="M22 15h2a3 3 0 0 1 0 6h-2"/><path d="M12 8c-1-1 1-2 0-3"/><path d="M16 8c-1-1 1-2 0-3"/><path d="M20 8c-1-1 1-2 0-3"/>'),
  headphones: iconSvg('<path d="M8 18a8 8 0 0 1 16 0"/><rect x="6" y="17" width="5" height="9" rx="2"/><rect x="21" y="17" width="5" height="9" rx="2"/><path d="M11 26h3"/><path d="M18 26h3"/>'),
  ereader: iconSvg('<rect x="9" y="5" width="14" height="22" rx="2"/><path d="M12 10h8"/><path d="M12 14h8"/><path d="M12 18h6"/><circle cx="16" cy="24" r="1"/>'),
  fax: iconSvg('<rect x="7" y="13" width="18" height="11" rx="2"/><path d="M11 13V6h10v7"/><path d="M11 20h10"/><path d="M11 24v3h10v-3"/><circle cx="22" cy="17" r="1"/>'),
  airFryer: iconSvg('<rect x="9" y="6" width="14" height="21" rx="4"/><path d="M12 15h8"/><path d="M12 19h8"/><path d="M13 11h6"/><path d="M23 14h2v6h-2"/>'),
  toothbrush: iconSvg('<path d="M11 26l8-18"/><path d="M18 8l4 2"/><path d="M20 5l4 2"/><path d="M10 24l5 2"/><path d="M22 7l1-3"/><path d="M19 6l1-3"/>'),
  steamCleaner: iconSvg('<path d="M9 25h14"/><path d="M12 25l4-15 4 15"/><path d="M13 10h6"/><path d="M15 7c-1-1 1-2 0-3"/><path d="M19 7c-1-1 1-2 0-3"/>'),
  chair: iconSvg('<path d="M11 6h9l2 10H10z"/><path d="M11 16v5h12"/><path d="M16 21v6"/><path d="M10 27h12"/><path d="M8 18h3"/>'),
  vr: iconSvg('<rect x="5" y="11" width="22" height="11" rx="4"/><circle cx="12" cy="17" r="2"/><circle cx="20" cy="17" r="2"/><path d="M27 15l3-2"/><path d="M5 15l-3-2"/>'),
  heatPump: iconSvg('<rect x="7" y="7" width="18" height="18" rx="3"/><circle cx="16" cy="16" r="5"/><path d="M16 11v10M11 16h10"/><path d="M8 26h16"/><path d="M10 4h12"/>'),
  phone: iconSvg('<rect x="9" y="12" width="14" height="13" rx="2"/><path d="M12 16h8"/><path d="M12 20h2M16 20h2M20 20h0"/><path d="M11 10c3-4 7-4 10 0"/>'),
  mower: iconSvg('<path d="M8 20h14l3-5h-8l-3 5"/><circle cx="10" cy="24" r="3"/><circle cx="22" cy="24" r="3"/><path d="M22 15l4-8"/><path d="M24 7h4"/>'),
  microwave: iconSvg('<rect x="5" y="9" width="22" height="15" rx="2"/><rect x="8" y="12" width="12" height="9" rx="1"/><path d="M23 12h1M23 16h1M23 20h1"/><path d="M10 17c2-2 5 2 8 0"/>'),
  motherboard: iconSvg('<rect x="6" y="6" width="20" height="20" rx="2"/><rect x="11" y="11" width="8" height="8" rx="1"/><path d="M9 15h2M19 15h4M15 9v2M15 19v4M22 9h1M9 23h1M23 23h1"/><path d="M19 19l4 4"/>'),
  robotVacuum: iconSvg('<circle cx="16" cy="16" r="10"/><circle cx="16" cy="16" r="4"/><path d="M16 6v4"/><path d="M9 23l-3 3"/><path d="M23 23l3 3"/>'),
  snowmobile: iconSvg('<path d="M7 22h17"/><path d="M10 18l4-6h7l4 6"/><path d="M13 12l-1-4h6"/><path d="M24 18l4 3"/><path d="M8 25h13"/>'),
  tv: iconSvg('<rect x="6" y="10" width="20" height="13" rx="2"/><path d="M13 26h6"/><path d="M16 23v3"/><path d="M12 7l4 3 4-3"/>'),
  walkieTalkie: iconSvg('<rect x="10" y="9" width="12" height="18" rx="2"/><path d="M14 9V5h4v4"/><path d="M13 14h6"/><path d="M13 18h6"/><circle cx="16" cy="23" r="1"/>'),
  fridge: iconSvg('<rect x="10" y="5" width="12" height="22" rx="2"/><path d="M10 14h12"/><path d="M14 10v2"/><path d="M14 18v2"/>'),
  oven: iconSvg('<rect x="7" y="7" width="18" height="20" rx="2"/><path d="M7 13h18"/><rect x="11" y="16" width="10" height="7" rx="1"/><path d="M12 10h1M16 10h1M20 10h1"/>'),
  thermostat: iconSvg('<circle cx="16" cy="16" r="10"/><circle cx="16" cy="16" r="4"/><path d="M16 16l5-4"/><path d="M16 7v2M25 16h-2M16 25v-2M7 16h2"/>'),
  mouse: iconSvg('<rect x="10" y="5" width="12" height="22" rx="6"/><path d="M16 5v7"/><path d="M16 12h6"/><path d="M7 11l-2 2 2 2"/><path d="M25 11l2 2-2 2"/>'),
  waterPump: iconSvg('<path d="M9 17h9l4 4v4H9z"/><path d="M18 17v-5h5"/><path d="M23 12v5"/><circle cx="13" cy="21" r="2"/><path d="M6 27c2-1 4-1 6 0s4 1 6 0 4-1 6 0"/>'),
  kidsBike: iconSvg('<circle cx="10" cy="23" r="3"/><circle cx="22" cy="23" r="3"/><path d="M10 23l5-8h5l2 8"/><path d="M15 15l-2-4h5"/><path d="M20 15l4-3"/><path d="M24 12h3"/>'),
  generator: iconSvg('<rect x="6" y="9" width="20" height="15" rx="3"/><path d="M11 9V6h10v3"/><path d="M17 12l-4 6h4l-2 5 5-7h-4z"/><circle cx="10" cy="25" r="1"/><circle cx="22" cy="25" r="1"/>'),
  keyboard: iconSvg('<rect x="5" y="10" width="22" height="14" rx="2"/><path d="M9 14h2M14 14h2M19 14h2M9 18h2M14 18h8M23 14h0"/>'),
  product: iconSvg('<path d="M8 11l8-5 8 5v10l-8 5-8-5z"/><path d="M8 11l8 5 8-5"/><path d="M16 16v10"/><path d="M12 8l8 5"/>'),
};

function iconSvg(content) {
  // `aria-hidden` keeps decorative product icons out of screen-reader output;
  // the product button already contains the product name as accessible text.
  return `<svg viewBox="0 0 32 32" aria-hidden="true" focusable="false">${content}</svg>`;
}

function productIconSvg(productName) {
  // Unknown product names fall back to a generic package icon rather than
  // showing broken markup.
  return SVG_ICONS[ICON_BY_PRODUCT[productName] || "product"];
}

function byProduct(product) {
  // `answers.json` stores a flat `items` array plus a product summary list. This
  // helper filters the flat array whenever a product is selected.
  return state.data.items.filter((item) => item.product === product);
}

function manualQuestionsForProduct(product) {
  if (!product) return [];
  return byProduct(product);
}

function pickRandom(list) {
  // Used for "换一个问题" and initial product selection. The list should always
  // be non-empty because it comes from `answers.json.products`.
  return list[Math.floor(Math.random() * list.length)];
}

function renderProducts(filterText = "") {
  // Rebuild the left product rail from data. Buttons are ordinary HTML buttons
  // so keyboard users can tab through products; each click fills a known sample
  // question for fast demos.
  els.productList.innerHTML = "";
  const normalizedFilter = String(filterText || "").trim().toLowerCase();
  const products = state.data.products.filter((product) => (
    !normalizedFilter || String(product.name || "").toLowerCase().includes(normalizedFilter)
  ));
  for (const product of products) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "product-item";
    button.dataset.product = product.name;
    button.classList.toggle("active", product.name === state.activeProduct);
    button.innerHTML = `
      <span class="product-icon">${productIconSvg(product.name)}</span>
      <span class="product-name">${escapeHtml(product.name)}</span>
      <span class="product-count">${product.count}</span>
    `;
    button.addEventListener("click", () => selectProduct(product.name));
    els.productList.appendChild(button);
  }
  if (!products.length) {
    const empty = document.createElement("div");
    empty.className = "product-list-empty";
    empty.textContent = "没有匹配的产品";
    els.productList.appendChild(empty);
  }
}

function selectProduct(productName) {
  // Product selection updates both state and the visible question box. It does
  // not focus the textarea, because automatic focus previously caused the whole
  // page to scroll and made the fixed layout look broken.
  state.activeProduct = productName;
  state.activeItem = pickRandom(byProduct(productName));
  els.activeProduct.textContent = `智能客服 · ${productName}快捷问题`;
  els.questionInput.value = cleanQuestion(state.activeItem.question);
  autoResizeInput();
  for (const btn of document.querySelectorAll(".product-item")) {
    btn.classList.toggle("active", btn.dataset.product === productName);
  }
  renderQuestionMenu();
  updateHistoryContextIndicator();
  closeQuestionMenu();
  if (window.matchMedia("(max-width: 820px)").matches) closeMobileDrawers();
}

function syncActiveProductScope(productName) {
  if (!productName) return;
  if (!knownProductNames().includes(productName)) {
    // A valid manual can exist without a dedicated quick-product category.
    // Clear the old highlight instead of making the previous product appear
    // to own the current answer.
    state.activeProduct = null;
    els.activeProduct.textContent = `智能客服 · ${productName}`;
    for (const btn of document.querySelectorAll(".product-item")) btn.classList.remove("active");
    updateHistoryContextIndicator("");
    return;
  }
  state.activeProduct = productName;
  els.activeProduct.textContent = `智能客服 · ${productName}快捷问题`;
  for (const btn of document.querySelectorAll(".product-item")) {
    btn.classList.toggle("active", btn.dataset.product === productName);
  }
  updateHistoryContextIndicator(productName);
}

function shuffleQuestion() {
  // Pick another sample question inside the current product bucket. If the page
  // somehow has no active product yet, initialize from the first product.
  if (!state.activeProduct) {
    selectProduct(state.data.products[0].name);
    return;
  }
  state.activeItem = pickRandom(byProduct(state.activeProduct));
  els.questionInput.value = cleanQuestion(state.activeItem.question);
  autoResizeInput();
  renderQuestionMenu();
}

function renderQuestionMenu() {
  // Build the recommended-question dropdown for the selected product. The
  // `客服售后` branch uses the 50 known service questions as template samples;
  // the submitted answer still goes through `/chat`, where the backend routes it
  // to either the customer-service API or the manual RAG backend.
  const menu = els.questionMenu;
  if (!menu || !state.data) return;
  const questions = manualQuestionsForProduct(state.activeProduct);
  menu.innerHTML = "";

  if (!questions.length) {
    const empty = document.createElement("div");
    empty.className = "question-menu-empty";
    empty.textContent = state.activeProduct === CUSTOMER_SERVICE_PRODUCT
      ? "客服售后当前使用 50 题模板样例"
      : "当前产品暂无推荐问题";
    menu.appendChild(empty);
    return;
  }

  const title = document.createElement("div");
  title.className = "question-menu-title";
  title.textContent = `${state.activeProduct}推荐问题`;
  menu.appendChild(title);

  for (const item of questions) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "question-option";
    option.classList.toggle("active", cleanQuestion(item.question) === els.questionInput.value);
    option.textContent = cleanQuestion(item.question);
    option.addEventListener("click", () => {
      state.activeItem = item;
      els.questionInput.value = cleanQuestion(item.question);
      autoResizeInput();
      renderQuestionMenu();
      closeQuestionMenu();
      els.questionInput.focus();
    });
    menu.appendChild(option);
  }
}

function toggleQuestionMenu() {
  // Keep the custom dropdown's ARIA state synchronized with visibility. This
  // matters because the page avoids a native `<select>` so long manual
  // questions can wrap naturally.
  if (!els.questionMenu || !els.questionMenuBtn) return;
  cancelQuestionMenuClose();
  if (els.questionMenu.hidden) {
    renderQuestionMenu();
    els.questionMenu.hidden = false;
    els.questionMenu.classList.remove("closing");
    els.questionMenuBtn.setAttribute("aria-expanded", "true");
  } else {
    closeQuestionMenu();
  }
}

function closeQuestionMenu() {
  // Close immediately and cancel delayed hover timers. This is used by explicit
  // menu toggles, Escape key handling, outside clicks and product switching.
  if (!els.questionMenu || !els.questionMenuBtn) return;
  cancelQuestionMenuClose();
  els.questionMenu.classList.remove("closing");
  els.questionMenu.hidden = true;
  els.questionMenuBtn.setAttribute("aria-expanded", "false");
}

function scheduleQuestionMenuClose() {
  // Delay hover-close so users can move between the textarea, menu button and
  // dropdown without losing the menu. The second timeout lets CSS play a short
  // closing animation before `hidden` removes it from layout.
  if (!els.questionMenu || els.questionMenu.hidden) return;
  cancelQuestionMenuClose();
  state.questionMenuCloseTimer = window.setTimeout(() => {
    if (!els.questionMenu || els.questionMenu.hidden) return;
    els.questionMenu.classList.add("closing");
    els.questionMenuBtn.setAttribute("aria-expanded", "false");
    state.questionMenuCloseTimer = window.setTimeout(() => {
      if (!els.questionMenu) return;
      els.questionMenu.hidden = true;
      els.questionMenu.classList.remove("closing");
      state.questionMenuCloseTimer = null;
    }, 220);
  }, 2000);
}

function cancelQuestionMenuClose() {
  // Shared reset for all paths that should keep the menu alive.
  if (state.questionMenuCloseTimer) {
    window.clearTimeout(state.questionMenuCloseTimer);
    state.questionMenuCloseTimer = null;
  }
  if (els.questionMenu) els.questionMenu.classList.remove("closing");
  if (els.questionMenuBtn && els.questionMenu && !els.questionMenu.hidden) {
    els.questionMenuBtn.setAttribute("aria-expanded", "true");
  }
}

function isInsideQuestionMenuArea(target) {
  // Treat the whole question-field wrapper as the active menu hover zone.
  return Boolean(target && els.questionField && els.questionField.contains(target));
}

function cleanQuestion(question) {
  // Clean only the display label used in the dropdown. The full original
  // question remains in `state.activeItem` and is what gets submitted to the
  // table/API path, preserving exact benchmark wording.
  return String(question || "")
    .replace(/\r?\n/g, " ")
    .replace(/"\s*,?\s*"/g, " ")
    .replace(/^"+|"+$/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function clearWelcome() {
  // The welcome splash is only a placeholder. The first user or assistant
  // message removes it so the conversation area becomes a normal chat transcript.
  const welcome = els.messages.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function formatExactTime(value) {
  const date = new Date(Number(value) || value);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (number) => String(number).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatThinkingTime(durationMs) {
  const seconds = Math.max(0, Math.round(Number(durationMs) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return `${minutes} 分 ${String(remainder).padStart(2, "0")} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${String(minutes % 60).padStart(2, "0")} 分 ${String(remainder).padStart(2, "0")} 秒`;
}

function createMessageMeta(className, text, timestamp = null) {
  const meta = document.createElement("div");
  meta.className = `message-meta ${className}`;
  meta.textContent = text;
  if (timestamp) meta.title = new Date(Number(timestamp) || timestamp).toISOString();
  return meta;
}

function addMessage(role, contentNode, meta = {}) {
  // Generic message renderer. `contentNode` can be plain text or a DOM node; the
  // assistant path uses DOM nodes so answers can contain manual image figures.
  clearWelcome();
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (typeof contentNode === "string") {
    bubble.textContent = contentNode;
  } else {
    bubble.appendChild(contentNode);
  }
  if (role === "assistant" && Number.isFinite(Number(meta.thinkingMs))) {
    wrap.appendChild(createMessageMeta(
      "thinking-time",
      `思考了 ${formatThinkingTime(meta.thinkingMs)} ›`,
    ));
  }
  wrap.appendChild(bubble);
  if (role === "user" && meta.askedAt) {
    wrap.appendChild(createMessageMeta(
      "question-time",
      `提问于 ${formatExactTime(meta.askedAt)}`,
      meta.askedAt,
    ));
  }
  if (role === "assistant" && meta.completedAt) {
    wrap.appendChild(createMessageMeta(
      "completed-time",
      `完整回答生成于 ${formatExactTime(meta.completedAt)}`,
      meta.completedAt,
    ));
  }
  els.messages.appendChild(wrap);
  els.messages.scrollTop = els.messages.scrollHeight;
  return wrap;
}

function renderAnswer(item) {
  // Convert an answer item from the API into the assistant bubble. The status line
  // deliberately hides implementation details
  // such as "matched Q67" because the public demo should feel like one coherent
  // intelligent customer-service response.
  const box = document.createElement("div");
  const status = document.createElement("div");
  status.className = "status-line";
  status.innerHTML = '<span class="dot"></span><span>智能体客服已经生成回复</span>';
  box.appendChild(status);

  const translationTools = document.createElement("div");
  translationTools.className = "answer-translation-tools";
  const translateButton = document.createElement("button");
  translateButton.type = "button";
  translateButton.className = "answer-translate-button";
  translateButton.textContent = "翻译当前答案";
  translateButton.title = "将当前答案及图片 caption 翻译成另一种语言";
  const translationStatus = document.createElement("span");
  translationStatus.className = "answer-translation-status";
  translationTools.append(translateButton, translationStatus);

  const answer = document.createElement("div");
  answer.className = "answer-text";
  if (item.imageDescriptions?.length) {
    answer.appendChild(createImageDescriptionPanel(item.imageDescriptions));
  }
  const answerContent = document.createElement("div");
  answerContent.className = "answer-content";
  renderAnswerWithImages(answerContent, normalizeFixedAnswerHeadingsForDisplay(item.answer, item), item.images, item.product);
  answer.appendChild(answerContent);
  box.appendChild(answer);

  translateButton.addEventListener("click", () => toggleAnswerTranslation(
    item,
    answerContent,
    translateButton,
    translationStatus,
  ));
  const modeTag = createAnswerModeTag(item);
  if (modeTag) translationTools.appendChild(modeTag);
  box.appendChild(translationTools);
  const confidenceTag = createAnswerConfidenceTag(item);
  if (confidenceTag) box.appendChild(confidenceTag);
  const sources = createSourcesPanel(item.sources, item.answer, item.images);
  if (sources) box.appendChild(sources);
  box.appendChild(createResolutionActions(item));
  return box;
}

function createResolutionActions(item) {
  const section = document.createElement("section");
  section.className = "resolution-actions";
  section.setAttribute("aria-label", "客服后续操作");
  section.innerHTML = `
    <div class="resolution-question">这次回答解决了您的问题吗？</div>
    <div class="resolution-buttons">
      <button type="button" class="resolution-button solved" data-resolution="solved">
        <span aria-hidden="true">✓</span>已解决
      </button>
      <button type="button" class="resolution-button unsolved" data-resolution="unsolved">
        <span aria-hidden="true">×</span>未解决
      </button>
      <button type="button" class="resolution-button transfer" data-service-shell="transfer">
        <span aria-hidden="true">↗</span>转人工客服
      </button>
      <button type="button" class="resolution-button ticket" data-service-shell="ticket">
        <span aria-hidden="true">▤</span>创建售后工单
      </button>
    </div>
    <div class="resolution-feedback" aria-live="polite"></div>
  `;
  section.addEventListener("click", (event) => {
    event.stopPropagation();
    const resolutionButton = event.target.closest("[data-resolution]");
    if (resolutionButton) {
      for (const button of section.querySelectorAll("[data-resolution]")) {
        button.classList.toggle("selected", button === resolutionButton);
      }
      const solved = resolutionButton.dataset.resolution === "solved";
      const feedback = section.querySelector(".resolution-feedback");
      feedback.className = `resolution-feedback ${solved ? "positive" : "attention"}`;
      feedback.textContent = solved
        ? "感谢反馈，已记录为已解决。"
        : "已记录为未解决，您可以继续选择人工客服或售后工单。";
      recordServiceAction(resolutionButton.dataset.resolution, item);
      return;
    }
    const shellButton = event.target.closest("[data-service-shell]");
    if (shellButton) {
      recordServiceAction(shellButton.dataset.serviceShell === "transfer" ? "transfer" : "ticket_open", item);
      openServiceShell(shellButton.dataset.serviceShell, item);
    }
  });
  return section;
}

async function recordServiceAction(action, item = {}) {
  if (!item.requestId) return;
  try {
    const response = await fetch(FEEDBACK_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Client-Type": "web-feedback" },
      body: JSON.stringify({
        request_id: item.requestId,
        product: item.product || "",
        action,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn("客服反馈记录失败", error);
  }
}

function closeServiceShell() {
  const modal = document.querySelector("#serviceShellModal");
  if (modal) modal.hidden = true;
}

function ensureServiceShellModal() {
  let modal = document.querySelector("#serviceShellModal");
  if (modal) return modal;
  modal = document.createElement("div");
  modal.id = "serviceShellModal";
  modal.className = "service-shell-modal";
  modal.hidden = true;
  modal.innerHTML = `
    <div class="service-shell-backdrop" data-service-shell-close="1"></div>
    <section class="service-shell-card" role="dialog" aria-modal="true" aria-labelledby="serviceShellTitle">
      <button type="button" class="service-shell-close" data-service-shell-close="1" aria-label="关闭">×</button>
      <div id="serviceShellContent"></div>
    </section>
  `;
  modal.addEventListener("click", (event) => {
    if (event.target.closest("[data-service-shell-close]")) closeServiceShell();
  });
  document.body.appendChild(modal);
  return modal;
}

function openServiceShell(kind, item = {}) {
  const modal = ensureServiceShellModal();
  const content = modal.querySelector("#serviceShellContent");
  const product = escapeHtml(item.product || "待识别产品");
  const question = escapeHtml(cleanQuestion(item.question || "当前咨询问题").slice(0, 160));
  if (kind === "ticket") {
    content.innerHTML = `
      <div class="service-shell-badge">演示模式</div>
      <h2 id="serviceShellTitle">创建售后工单</h2>
      <p class="service-shell-intro">当前仅展示工单交互壳，暂未连接真实售后工单系统。</p>
      <form id="ticketShellForm" class="ticket-shell-form">
        <label><span>关联产品</span><input value="${product}" readonly></label>
        <label><span>问题摘要</span><textarea rows="3">${question}</textarea></label>
        <label><span>联系方式</span><input placeholder="手机号或邮箱（演示中不会提交）"></label>
        <div class="service-shell-actions">
          <button type="button" class="shell-secondary" data-service-shell-close="1">取消</button>
          <button type="submit" class="shell-primary">确认创建（演示）</button>
        </div>
      </form>
      <div id="ticketShellResult" class="service-shell-result" hidden></div>
    `;
    content.querySelector("#ticketShellForm").addEventListener("submit", (event) => {
      event.preventDefault();
      recordServiceAction("ticket_submit", item);
      const result = content.querySelector("#ticketShellResult");
      result.hidden = false;
      result.textContent = "工单交互壳运行正常；本次未向任何外部系统提交数据。";
      event.currentTarget.querySelector(".shell-primary").disabled = true;
    });
  } else {
    content.innerHTML = `
      <div class="service-shell-badge">演示模式</div>
      <h2 id="serviceShellTitle">转接人工客服</h2>
      <p class="service-shell-intro">未来接入真实坐席后，将自动携带本次问题、AI 回答、产品信息和检索证据。</p>
      <div class="transfer-preview">
        <div><span>转接状态</span><strong>人工坐席接口待接入</strong></div>
        <div><span>关联产品</span><strong>${product}</strong></div>
        <div><span>问题摘要</span><strong>${question}</strong></div>
      </div>
      <div class="service-shell-notice">当前版本不会排队、发送消息或创建真实客服会话。</div>
      <div class="service-shell-actions">
        <button type="button" class="shell-primary" data-service-shell-close="1">知道了</button>
      </div>
    `;
  }
  modal.hidden = false;
  window.setTimeout(() => modal.querySelector(".service-shell-close")?.focus(), 20);
}

function splitTranslationLine(line) {
  const text = String(line || "").trim();
  if (!text) return [];
  const segments = [];
  let start = 0;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const isCjkBoundary = "。！？!?；;".includes(char);
    const isEnglishBoundary = ".!?".includes(char)
      && (index === text.length - 1 || /\s/.test(text[index + 1]))
      && !(char === "." && index <= 2 && /^\d+\./.test(text));
    if (!isCjkBoundary && !isEnglishBoundary) continue;
    const part = text.slice(start, index + 1).trim();
    if (part) segments.push(part);
    start = index + 1;
    while (/\s/.test(text[start] || "")) start += 1;
    index = start - 1;
  }
  const rest = text.slice(start).trim();
  if (rest) segments.push(rest);
  return segments;
}

function splitTranslationSegments(answer) {
  const segments = [];
  let pendingPics = 0;
  for (const part of String(answer || "").split(/(<PIC(?::[^>]+)?>)/g)) {
    if (/^<PIC(?::[^>]+)?>$/.test(part)) {
      if (segments.length) segments[segments.length - 1].picCount += 1;
      else pendingPics += 1;
      continue;
    }
    for (const line of part.replace(/\r\n/g, "\n").split(/\n+/)) {
      for (const text of splitTranslationLine(line)) {
        segments.push({ text, picCount: pendingPics });
        pendingPics = 0;
      }
    }
  }
  if (pendingPics && segments.length) segments[segments.length - 1].picCount += pendingPics;
  return segments;
}

function isEnglishQuestion(question) {
  const text = String(question || "").trim();
  if (!text) return false;
  const latinCharacters = (text.match(/[A-Za-z]/g) || []).length;
  const cjkCharacters = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const englishWords = text.match(/[A-Za-z]+(?:['-][A-Za-z]+)*/g) || [];
  // Require both an English-looking phrase and a clear Latin-script majority.
  // This avoids translating Chinese questions that merely contain a model name
  // such as DCD791 or an English button label.
  return englishWords.length >= 2 && latinCharacters >= 6 && latinCharacters > cjkCharacters * 2;
}

function isEnglishAnswer(answer) {
  const text = String(answer || "");
  const latinCharacters = (text.match(/[A-Za-z]/g) || []).length;
  const cjkCharacters = (text.match(/[\u3400-\u9fff]/g) || []).length;
  return latinCharacters >= 6 && latinCharacters > cjkCharacters * 2;
}

async function prepareEnglishAnswerTranslation(item) {
  const answer = String(item?.answer || "");
  // English questions keep the English answer as the primary display. Only
  // translate when the returned answer is actually English; the result is the
  // Chinese copy shown underneath it.
  if (!isEnglishAnswer(answer)) return null;
  const segments = splitTranslationSegments(answer);
  if (!segments.length) return null;
  const translations = await translateSegments(segments, "gpt-5.6-luna");
  return { segments, translations };
}

async function prepareEnglishQuestionTranslation(question) {
  if (!isEnglishQuestion(question)) return null;
  const segment = { text: String(question).trim(), picCount: 0 };
  const translations = await translateSegments([segment], "gpt-5.6-luna");
  return translations[0] || "";
}

function appendInlineQuestionTranslation(userWrap, question, translated) {
  const text = String(translated || "").trim();
  const bubble = userWrap?.querySelector(".bubble");
  if (!text || !bubble) return;
  const original = document.createElement("div");
  original.textContent = question;
  const translation = document.createElement("div");
  translation.className = "question-translation";
  translation.textContent = text;
  bubble.textContent = "";
  bubble.append(original, translation);
}

function replaceWithInlineChineseTranslation(target, item, translation) {
  if (!target || !translation?.segments?.length) return;
  // Match the manual viewer's bilingual layout: every original English block
  // stays in place and its green Chinese translation is inserted immediately
  // below it.  This runs only after the full English answer is already visible.
  renderTranslatedAnswer(target, item, translation.segments, translation.translations, {});
  els.messages.scrollTop = els.messages.scrollHeight;
}

async function translateSegments(segments, model) {
  const response = await fetch(TRANSLATE_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments: segments.map((segment) => segment.text),
      model: model || state.activeModelProfile?.model || "gpt-5.6-terra",
    }),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok || payload?.code !== 0) {
    throw new Error(payload?.detail || payload?.msg || "翻译请求失败");
  }
  const translations = payload.data?.translations;
  if (!Array.isArray(translations) || translations.length !== segments.length) {
    throw new Error("翻译结果与原文句子数量不一致");
  }
  return translations.map((text) => String(text || "").trim());
}

function renderTranslatedAnswer(target, item, answerSegments, translations, captionData) {
  target.innerHTML = "";
  let imageIndex = 0;
  answerSegments.forEach((segment, index) => {
    const row = document.createElement("div");
    row.className = "answer-translation-row";
    const original = document.createElement("div");
    original.className = "answer-translation-original";
    renderRichAnswerText(original, segment.text);
    const translated = document.createElement("div");
    translated.className = "answer-translation";
    translated.textContent = translations[index] || "";
    row.append(original, translated);
    target.appendChild(row);

    for (let count = 0; count < segment.picCount && imageIndex < (item.images || []).length; count += 1) {
      const image = item.images[imageIndex];
      const caption = captionData[image.name]?.caption || "";
      const captionTranslation = captionData[image.name]?.translation || "";
      target.appendChild(createImageFigure(image, item.product, caption, captionTranslation));
      imageIndex += 1;
    }
  });
  while (imageIndex < (item.images || []).length) {
    const image = item.images[imageIndex];
    const caption = captionData[image.name]?.caption || "";
    const captionTranslation = captionData[image.name]?.translation || "";
    target.appendChild(createImageFigure(image, item.product, caption, captionTranslation));
    imageIndex += 1;
  }
}

async function toggleAnswerTranslation(item, target, button, status) {
  if (button.dataset.translated === "true") {
    target.innerHTML = "";
    renderAnswerWithImages(target, item.answer, item.images, item.product);
    button.dataset.translated = "false";
    button.textContent = "翻译当前答案";
    status.textContent = "";
    return;
  }
  button.disabled = true;
  status.textContent = "正在翻译...";
  try {
    const answerSegments = splitTranslationSegments(item.answer);
    const captionEntries = await Promise.all((item.images || []).map(async (image) => ({
      image,
      caption: await getImageCaption(item.product, image.name),
    })));
    const captionSegments = captionEntries
      .filter((entry) => entry.caption)
      .map((entry) => ({ text: entry.caption, imageName: entry.image.name }));
    const translations = await translateSegments(
      [...answerSegments, ...captionSegments],
      state.activeModelProfile?.model,
    );
    const captionData = {};
    captionSegments.forEach((entry, index) => {
      captionData[entry.imageName] = {
        caption: entry.text,
        translation: translations[answerSegments.length + index],
      };
    });
    renderTranslatedAnswer(
      target,
      item,
      answerSegments,
      translations.slice(0, answerSegments.length),
      captionData,
    );
    button.dataset.translated = "true";
    button.textContent = "显示原文";
    status.textContent = "已翻译";
  } catch (error) {
    console.warn(error);
    status.textContent = error?.message || "翻译失败";
  } finally {
    button.disabled = false;
  }
}

function answerRelatedSources(sources, answer) {
  const candidates = Array.isArray(sources)
    ? sources.filter((source) => source?.chunk_id)
    : [];
  // A multi-intent answer cites several distinct chunks/sections. Previously
  // this kept only the single `primary_evidence` source (or the first one),
  // so every non-primary intent's evidence became invisible and un-clickable.
  // Surface all recalled chunks, de-duplicated by chunk_id, with the primary
  // anchor(s) on top and the rest kept in document order for stable reading.
  const seen = new Set();
  const deduped = candidates.filter((source) => {
    const key = String(source?.chunk_id || "");
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!deduped.length) return [];
  const orderValue = (source) => {
    const raw = source?.document_order;
    return Number.isFinite(raw) ? raw : (Number.isFinite(source?.rank) ? source.rank : Number.MAX_SAFE_INTEGER);
  };
  return deduped.slice().sort((left, right) => (
    (right.primary_evidence ? 1 : 0) - (left.primary_evidence ? 1 : 0)
    || orderValue(left) - orderValue(right)
  ));
}

function createSourcesPanel(sources, answer, images = []) {
  const items = answerRelatedSources(sources, answer);
  const relatedSources = Array.isArray(sources)
    ? sources.filter((source) => source?.chunk_id)
    : [];
  if (!items.length) return null;
  const details = document.createElement("details");
  details.className = "answer-sources";
  const summary = document.createElement("summary");
  summary.innerHTML = `<span class="source-icon" aria-hidden="true">↗</span><span>来源</span><span class="source-count">${items.length}</span>`;
  details.appendChild(summary);
  const list = document.createElement("div");
  list.className = "source-list";
  items.forEach((source, index) => {
    const article = document.createElement("article");
    article.className = "source-item";
    article.tabIndex = 0;
    article.setAttribute("role", "button");
    article.setAttribute("aria-label", `查看来源 ${index + 1} 的完整信息`);
    const heading = document.createElement("div");
    heading.className = "source-heading";
    heading.textContent = `${index + 1}. ${source.section || source.manual || `召回片段 ${index + 1}`}`;
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = [source.manual, source.page != null ? `第 ${source.page} 页` : "", source.chunk_id]
      .filter(Boolean).join(" · ");
    article.append(heading, meta);
    if (source.excerpt) {
      const excerpt = document.createElement("p");
      excerpt.textContent = source.excerpt;
      article.appendChild(excerpt);
    }
    article.addEventListener("click", () => openSourcePopover(source, article, answer, images, relatedSources));
    article.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openSourcePopover(source, article, answer, images, relatedSources);
      }
    });
    list.appendChild(article);
  });
  details.appendChild(list);
  return details;
}

let sourcePopoverPositionCleanup = null;

function closeSourcePopover() {
  sourcePopoverPositionCleanup?.();
  sourcePopoverPositionCleanup = null;
  document.querySelector(".source-popover")?.remove();
}

function manualSourceUrl(source, answer = "", images = [], relatedSources = []) {
  // Resolve by semantic evidence instead of a fragile page number, so source
  // links remain useful after a manual is re-indexed or its pagination shifts.
  // The cited chunks narrow the search area; the final answer still decides
  // which individual manual lines receive a highlight inside each area.
  const url = new URL(MANUAL_INDEX_ENDPOINT, window.location.origin);
  const locator = new URLSearchParams();
  const manual = String(source?.manual || "").trim();
  url.searchParams.set("manual", manual);
  locator.set("chunk", String(source?.chunk_id || ""));

  // A source-card click is an explicit request to locate that one citation.
  // Passing every sibling source made the navigator highlight all of them and
  // scroll to the earliest document position, which could open source #1 when
  // the user clicked source #2.
  const answerImageNames = [...new Set((images || [])
    .map((image) => typeof image === "string" ? image : image?.name)
    .filter(Boolean))];
  const clickedKey = String(source?.group_id || `${manual}\u0000${String(source?.section || "").trim()}`);
  const seenChunks = new Set();
  const groupByKey = new Map();
  const groupOrder = [];
  for (const candidate of [source]) {
    if (!candidate?.chunk_id) continue;
    if (String(candidate.manual || "").trim() !== manual) continue; // same manual only
    const chunkId = String(candidate.chunk_id);
    if (seenChunks.has(chunkId)) continue;
    seenChunks.add(chunkId);
    const key = String(candidate.group_id || `${manual}\u0000${String(candidate.section || "").trim()}`);
    if (!groupByKey.has(key)) {
      groupByKey.set(key, { key, section: String(candidate.section || ""), texts: [], pics: new Set(), order: candidate.document_order, isClicked: false });
      groupOrder.push(key);
    }
    const group = groupByKey.get(key);
    const text = String(candidate.content || candidate.excerpt || "").trim();
    if (text) group.texts.push(text);
    for (const pic of text.matchAll(/\[\[PIC:([^\]]+)\]\]/g)) {
      const name = String(pic[1] || "").trim();
      if (name) group.pics.add(name);
    }
    if (key === clickedKey) group.isClicked = true;
    if (group.order == null && candidate.document_order != null) group.order = candidate.document_order;
  }

  // Answer figures belong to the section whose chunk actually cites them
  // (`[[PIC:...]]` anchors survive in the projected source content). Assign
  // each figure to its owning group so a multi-intent answer highlights the
  // replacement-filter figure inside the replacement section instead of
  // painting every figure onto the clicked section only. Figures no group
  // claims keep the legacy fallback: they land on the clicked group.
  const unclaimedImages = answerImageNames.filter((name) => (
    ![...groupByKey.values()].some((group) => group.pics.has(name))
  ));
  const orderValue = (group) => (Number.isFinite(group.order) ? group.order : Number.MAX_SAFE_INTEGER);
  const groups = groupOrder
    .map((key) => groupByKey.get(key))
    .sort((left, right) => orderValue(left) - orderValue(right))
    .map((group) => ({
      section: group.section,
      // Combine adjacent sliding-window chunks from this exact subsection to
      // complete a cited area without leaking into neighboring sections.
      excerpt: group.texts.join("\n\n").slice(0, 12000),
      pics: answerImageNames.filter((name) => group.pics.has(name)
        || (group.isClicked && unclaimedImages.includes(name))),
      isClicked: group.isClicked,
    }))
    .filter((group) => group.section || group.excerpt);

  // `answer` is still the highlight authority inside each resolved section.
  locator.set("answer", String(answer || "").slice(0, 12000));
  if (groups.length) {
    locator.set("groups", JSON.stringify(groups));
  }
  // Legacy single-group params keep old navigator builds working unchanged.
  // They must describe the CLICKED group, not simply the first group that
  // happens to own a figure.
  const clicked = groups.find((group) => group.isClicked) || groups[0] || { section: String(source?.section || ""), excerpt: String(answer || "") };
  locator.set("section", String(clicked.section || ""));
  locator.set("excerpt", String(clicked.excerpt || answer || "").slice(0, 12000));
  if (answerImageNames.length) locator.set("pics", answerImageNames.join(","));
  url.hash = locator.toString();
  return url.toString();
}

function positionSourcePopover(popover, anchor) {
  if (!popover?.isConnected || !anchor?.isConnected) return;
  const visualViewport = window.visualViewport;
  const viewport = {
    left: visualViewport?.offsetLeft || 0,
    top: visualViewport?.offsetTop || 0,
    width: visualViewport?.width || document.documentElement.clientWidth || window.innerWidth,
    height: visualViewport?.height || document.documentElement.clientHeight || window.innerHeight,
  };
  viewport.right = viewport.left + viewport.width;
  viewport.bottom = viewport.top + viewport.height;
  const anchorRect = anchor.getBoundingClientRect();
  const gap = 12;
  const margin = Math.max(8, Math.min(14, viewport.width * 0.012));
  const availableWidth = Math.max(0, viewport.width - margin * 2);
  const preferredWidth = Math.min(480, Math.max(320, viewport.width * 0.36));
  const rightSpace = viewport.right - anchorRect.right - gap - margin;
  const leftSpace = anchorRect.left - viewport.left - gap - margin;
  const minimumSideWidth = Math.min(300, preferredWidth, availableWidth);
  let width;
  let left;

  if (rightSpace >= minimumSideWidth) {
    width = Math.min(preferredWidth, rightSpace);
    left = anchorRect.right + gap;
  } else if (leftSpace >= minimumSideWidth) {
    width = Math.min(preferredWidth, leftSpace);
    left = anchorRect.left - width - gap;
  } else {
    width = Math.min(480, availableWidth);
    left = Math.max(viewport.left + margin, Math.min(anchorRect.left, viewport.right - width - margin));
  }

  width = Math.max(0, Math.min(width, availableWidth));
  left = Math.max(viewport.left + margin, Math.min(left, viewport.right - width - margin));
  popover.style.width = `${Math.floor(width)}px`;
  popover.style.left = `${Math.round(left)}px`;

  // Width changes wrapping and therefore height. Measure only after the final
  // width is applied, then give the middle content pane the remaining height.
  popover.style.height = "auto";
  popover.style.maxHeight = "none";
  const maximumHeight = Math.max(0, Math.min(720, viewport.height - margin * 2));
  const naturalHeight = popover.getBoundingClientRect().height;
  const finalHeight = Math.min(naturalHeight, maximumHeight);
  popover.style.height = `${Math.floor(finalHeight)}px`;
  popover.style.maxHeight = `${Math.floor(maximumHeight)}px`;
  const renderedHeight = popover.getBoundingClientRect().height;
  const maximumTop = viewport.bottom - margin - renderedHeight;
  const top = Math.max(viewport.top + margin, Math.min(anchorRect.top, maximumTop));
  popover.style.top = `${Math.round(top)}px`;
}

function trackSourcePopoverPosition(popover, anchor) {
  let frame = 0;
  const reposition = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      frame = 0;
      positionSourcePopover(popover, anchor);
    });
  };
  window.addEventListener("resize", reposition);
  document.addEventListener("scroll", reposition, true);
  window.visualViewport?.addEventListener("resize", reposition);
  window.visualViewport?.addEventListener("scroll", reposition);
  reposition();
  return () => {
    if (frame) cancelAnimationFrame(frame);
    window.removeEventListener("resize", reposition);
    document.removeEventListener("scroll", reposition, true);
    window.visualViewport?.removeEventListener("resize", reposition);
    window.visualViewport?.removeEventListener("scroll", reposition);
  };
}

function openSourcePopover(source, anchor, answer = "", images = [], relatedSources = []) {
  closeSourcePopover();
  const popover = document.createElement("aside");
  popover.className = "source-popover";
  popover.setAttribute("role", "dialog");
  popover.setAttribute("aria-label", "Chunk 完整信息");
  const header = document.createElement("div");
  header.className = "source-popover-header";
  const title = document.createElement("div");
  title.className = "source-popover-title";
  title.textContent = source.section || source.manual || "Chunk 完整信息";
  const close = document.createElement("button");
  close.className = "source-popover-close";
  close.type = "button";
  close.title = "关闭";
  close.setAttribute("aria-label", "关闭 Chunk 完整信息");
  close.textContent = "×";
  close.addEventListener("click", closeSourcePopover);
  header.append(title, close);
  const meta = document.createElement("div");
  meta.className = "source-popover-meta";
  meta.textContent = [source.manual, source.page != null ? `第 ${source.page} 页` : "", source.chunk_id]
    .filter(Boolean).join(" · ");
  const content = document.createElement("div");
  content.className = "source-popover-content";
  content.textContent = source.content || source.excerpt || "暂无更多正文信息";
  const locate = document.createElement("button");
  locate.className = "source-popover-locate";
  locate.type = "button";
  locate.textContent = "在手册中定位";
  locate.title = "在公开手册目录中打开并定位到该来源章节";
  locate.addEventListener("click", () => window.open(manualSourceUrl(source, answer, images, relatedSources), "_blank", "noopener"));
  const footer = document.createElement("div");
  footer.className = "source-popover-footer";
  footer.appendChild(locate);
  popover.append(header, meta, content, footer);
  document.body.appendChild(popover);
  positionSourcePopover(popover, anchor);
  sourcePopoverPositionCleanup = trackSourcePopoverPosition(popover, anchor);
  close.focus();
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeSourcePopover();
});

function createAnswerModeTag(item) {
  // Show the backend's final routing decision under every assistant answer.
  // Customer-service answers use a green tag; manual RAG answers use a yellow
  // tag so testers can immediately see which backend handled a mixed question.
  const mode = item.answerMode || (item.source === "customer-api" ? "customer" : "manual");
  if (!["customer", "manual"].includes(mode)) return null;
  const tag = document.createElement("div");
  tag.className = `answer-mode-tag ${mode === "customer" ? "customer" : "manual"}`;
  tag.textContent = mode === "customer" ? "客服模式" : "手册模式";
  return tag;
}

function confidenceBigrams(text) {
  const normalized = String(text || "").toLowerCase().replace(/\s+/g, "");
  const tokens = new Set(normalized.match(/[a-z0-9][a-z0-9._%-]+/g) || []);
  const cjk = normalized.replace(/[^\u3400-\u9fff]/g, "");
  for (let index = 0; index < cjk.length - 1; index += 1) tokens.add(cjk.slice(index, index + 2));
  return tokens;
}

function calculateAnswerConfidence(item) {
  if ((item.answerMode || "manual") !== "manual") return null;
  const backend = item.answerConfidence || item.retrievalTrace?.answer_confidence;
  if (backend?.level && Number.isFinite(Number(backend.score))) {
    const tone = { high: "high", medium: "medium", low: "low" }[backend.level] || "low";
    const level = { high: "高", medium: "中", low: "低" }[backend.level] || "低";
    return { score: Number(backend.score), level, tone, action: backend.action || "refuse" };
  }
  const sources = Array.isArray(item.sources) ? item.sources.filter(source => source?.chunk_id) : [];
  if (!sources.length) return { score: 25, level: "低", tone: "low" };
  const evidenceText = sources.map(source => source.content || source.excerpt || "").join("\n");
  const evidenceTokens = confidenceBigrams(evidenceText);
  const sentences = String(item.answer || "")
    .split(/[。！？!?\n]+/)
    .map(sentence => sentence.trim())
    .filter(sentence => sentence.length >= 8);
  let supported = 0;
  for (const sentence of sentences) {
    const tokens = confidenceBigrams(sentence);
    if (!tokens.size) continue;
    let matches = 0;
    tokens.forEach(token => { if (evidenceTokens.has(token)) matches += 1; });
    if (matches / tokens.size >= 0.34) supported += 1;
  }
  const coverage = sentences.length ? supported / sentences.length : 0;
  const sourceStrength = Math.min(sources.length, 3) / 3;
  const hasFullContent = sources.some(source => String(source.content || "").length > String(source.excerpt || "").length);
  const hasImages = Array.isArray(item.images) && item.images.length > 0;
  const raw = 25 + coverage * 55 + sourceStrength * 10 + (hasFullContent ? 5 : 0) + (hasImages ? 5 : 0);
  const score = Math.max(25, Math.min(95, Math.round(raw)));
  if (score >= 80) return { score, level: "高", tone: "high" };
  if (score >= 60) return { score, level: "中", tone: "medium" };
  return { score, level: "低", tone: "low" };
}

function createAnswerConfidenceTag(item) {
  const confidence = calculateAnswerConfidence(item);
  if (!confidence) return null;
  const tag = document.createElement("div");
  tag.className = `answer-confidence ${confidence.tone}`;
  tag.title = "后端检索完成后统一决策：高置信回答，中置信澄清，低置信拒答。";
  const action = { answer: "已回答", clarify: "已澄清", refuse: "已拒答" }[confidence.action] || "已回答";
  tag.innerHTML = `<span class="confidence-dot" aria-hidden="true"></span><span>回答置信度</span><strong>${confidence.score}%</strong><span>${confidence.level} · ${action}</span>`;
  return tag;
}

function errorNode(error) {
  const box = document.createElement("div");
  const status = document.createElement("div");
  status.className = "status-line";
  status.innerHTML = '<span class="dot"></span><span>接口调用失败</span>';
  const message = document.createElement("div");
  message.className = "answer-text";
  const text = error?.message || "请稍后重试，或检查后端 /chat 服务状态。";
  message.textContent = text;
  box.append(status, message);
  return box;
}

function createImageDescriptionPanel(descriptions) {
  // Surface the dedicated vision pre-parser result so demos can confirm the
  // uploaded image was actually interpreted before the manual RAG answer.
  const panel = document.createElement("div");
  panel.className = "image-description-panel";
  const title = document.createElement("div");
  title.className = "image-description-title";
  title.textContent = "上传图片解析结果";
  panel.appendChild(title);

  const list = document.createElement("ol");
  list.className = "image-description-list";
  descriptions.forEach((description) => {
    const li = document.createElement("li");
    li.textContent = description;
    list.appendChild(li);
  });
  panel.appendChild(list);
  return panel;
}

function renderAnswerWithImages(container, answer, images, product) {
  // The backend uses `<PIC>` placeholders to mark where manual images belong.
  // We split on that marker and interleave the image queue. Any leftover images
  // render as a compact grid so useful figures are still visible even if the
  // text has fewer placeholders than image names.
  const source = String(answer || "");
  const imgQueue = [...(images || [])];
  const trailing = trailingPicRun(source);
  if (trailing && trailing.count > 1 && imgQueue.length) {
    renderTrailingPicsNearText(container, trailing.text, imgQueue, product);
    return;
  }

  const parts = source.split(/<PIC(?::[^>]+)?>/g);
  parts.forEach((part, index) => {
    const text = part.trim();
    if (text) {
      renderRichAnswerText(container, text);
    }
    if (index < parts.length - 1 && imgQueue.length) {
      const img = imgQueue.shift();
      container.appendChild(createImageFigure(img, product));
    }
  });
  if (imgQueue.length) {
    const grid = document.createElement("div");
    grid.className = "image-grid inline-image-grid";
    imgQueue.forEach((img) => grid.appendChild(createImageFigure(img, product)));
    container.appendChild(grid);
  }
}

// The server performs the same normalization when it loads the recommendation
// table. This browser-side mirror also protects an already-running gateway
// whose in-memory cache predates a table refresh. It only inserts line breaks
// at fixed manual heading boundaries; it never changes prose or `<PIC>`.
function normalizeFixedAnswerHeadingsForDisplay(answer, item = {}) {
  if ((item.answerMode || "manual") !== "manual" || item.product === "客服售后") return String(answer || "");
  const source = String(answer || "");
  const markers = [...source.matchAll(/(?<!#)(#{1,6}) +/g)];
  const points = new Set();
  const titleBoundary = (fragment) => {
    const line = String(fragment || "").split("\n", 1)[0];
    const repeated = line.match(/^(.{2,36}?) \1(?=\S)/u);
    if (repeated) return repeated[1].length;
    const named = line.match(/^(.{2,48}?(?:功能|说明|数据|操作|设置|安装|维护|清洁|保养|调节|调整|运行|模式|显示|按钮|部件|组件|装备|安全|警告|注意事项|概览|规格|步骤|存放|连接|排水|洗涤剂|洗涤块|亮碟剂|滤网|系统|程序|电池|温度|餐具|物品|建议|高度|停机|介绍|使用|检查|更换|拆卸|组装|充电|开机|关机|故障排除)) (?=\S)/u);
    if (named) return named[1].length;
    const subject = line.match(/^(.{2,64}?) (?=(?:本|该|此|这|机器|产品|设备|洗碗机|空调|冰箱|健身单车|健身追踪器|控制台|新产品|本机|本产品|本设备|该机|该产品|该设备|您的|你(?:的)?))/u);
    if (subject) return subject[1].length;
    const direct = line.match(/^(.{2,44}?) (?=(?:通过|使用|按(?:下)?|请|将|可|需|为|在|从|要|如|若|当|对于|以下|图|[0-9]+[.、]|[-•*]|<PIC>))/u);
    if (direct) return direct[1].length;
    const english = line.match(/^([A-Z][A-Za-z0-9 /&'()_-]{2,64}?) (?=(?:The|This|Your|To|When|If|Before|After|Use|Check|Press|Remove|Install|Do|Never|Push|As|One|Severe|Risk|<PIC>|[0-9]+\.))/);
    return english ? english[1].length : null;
  };
  markers.forEach((marker, index) => {
    const start = marker.index || 0;
    const end = start + marker[0].length;
    if (start > 0 && source[start - 1] !== "\n") points.add(start);
    const next = index + 1 < markers.length ? markers[index + 1].index : source.length;
    const boundary = titleBoundary(source.slice(end, next));
    if (boundary !== null && source[end + boundary] !== "\n") points.add(end + boundary);
  });
  if (!points.size) return source;
  let output = "";
  let cursor = 0;
  for (const point of [...points].sort((left, right) => left - right)) {
    output += source.slice(cursor, point) + "\n";
    cursor = point;
  }
  return output + source.slice(cursor);
}

function trailingPicRun(answer) {
  // Some compact RAG repairs can only recover the right image set, so they append
  // a run of `<PIC>` markers at the end. For the web UI, distribute that run
  // back across nearby paragraphs/steps instead of showing every figure below
  // the whole answer.
  const text = String(answer || "");
  const match = text.match(/(?:\s*<PIC(?::[^>]+)?>\s*)+$/);
  if (!match) return null;
  const body = text.slice(0, match.index).trim();
  if (!body || /<PIC(?::[^>]+)?>/.test(body)) return null;
  const count = (match[0].match(/<PIC(?::[^>]+)?>/g) || []).length;
  return { text: body, count };
}

function renderTrailingPicsNearText(container, answer, images, product) {
  const units = splitAnswerForImageDistribution(answer);
  if (!units.length) {
    images.forEach((img) => container.appendChild(createImageFigure(img, product)));
    return;
  }

  const positions = distributedImagePositions(units.length, images.length);
  const grouped = new Map();
  images.forEach((img, index) => {
    const pos = positions[index];
    if (!grouped.has(pos)) grouped.set(pos, []);
    grouped.get(pos).push(img);
  });

  units.forEach((unit, index) => {
    renderRichAnswerText(container, unit);
    const group = grouped.get(index) || [];
    group.forEach((img) => container.appendChild(createImageFigure(img, product)));
  });
}

function splitAnswerForImageDistribution(answer) {
  const lines = String(answer || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const numberedCount = lines.filter((line) => /^\d+[.)]\s+/.test(line)).length;
  if (numberedCount >= 2) {
    const units = [];
    let current = [];
    for (const line of lines) {
      const startsNumbered = /^\d+[.)]\s+/.test(line);
      if (startsNumbered && current.length) {
        units.push(current.join("\n"));
        current = [];
      }
      current.push(line);
    }
    if (current.length) units.push(current.join("\n"));
    return units;
  }

  const paragraphs = String(answer || "")
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (paragraphs.length > 1) return paragraphs;

  return lines.length ? lines : [String(answer || "").trim()].filter(Boolean);
}

function distributedImagePositions(unitCount, imageCount) {
  if (unitCount <= 0 || imageCount <= 0) return [];
  if (imageCount <= unitCount) {
    const used = new Set();
    return Array.from({ length: imageCount }, (_, index) => {
      const raw = Math.round(((index + 1) * unitCount) / imageCount) - 1;
      let pos = Math.min(unitCount - 1, Math.max(0, raw));
      while (used.has(pos) && pos + 1 < unitCount) pos += 1;
      while (used.has(pos) && pos > 0) pos -= 1;
      used.add(pos);
      return pos;
    });
  }
  return Array.from({ length: imageCount }, (_, index) => Math.min(unitCount - 1, index));
}

function renderRichAnswerText(container, text) {
  // Render model/table answers as structured blocks. Multi-question answers are
  // easier to judge when headings, bullets and paragraphs remain visually
  // separate instead of becoming one dense text node.
  const blocks = splitAnswerBlocks(text);
  for (const block of blocks) {
    const node = createAnswerBlock(block);
    if (node) container.appendChild(node);
  }
}

function isLikelyAnswerSubheading(line, hasFollowingLine = true) {
  const value = String(line || "").trim();
  if (!value || value.length > 56) return false;
  // A `#` token is a heading only when data has already made it a standalone
  // line.  Treating `# 标题 正文…` as a heading made the complete paragraph
  // bold whenever an OCR-flattened fixed answer reached the browser.
  if (/^#{1,6}\s+\S/.test(value)) {
    const title = value.replace(/^#{1,6}\s+/, "");
    // A short flattened source line can otherwise evade the length guard,
    // e.g. `# 清洁前 务必先断电`. It is safer to show that as ordinary
    // text than to bold its entire body. The cache normalizer supplies the
    // valid `# 标题\n正文` form used by reviewed answers.
    const looksLikeInlineBody = /(?:[。！？.!?]\s+|\s+(?:本|该|此|这|机器|产品|设备|洗碗机|空调|冰箱|控制台|软水系统|通常|务必|请|可|需|会|应|将|使用|检查|清洁|显示|When|If|The|This|Your|To|Do|Never|Press|Use|Check|Install|Remove))/i.test(title);
    return hasFollowingLine && !looksLikeInlineBody;
  }
  if (!hasFollowingLine) return false;
  if (/^关于[“\"].*?[”\"]的问题，回答如下：$/.test(value)) return false;
  if (/^\*\*[^*]+\*\*[：:。.]?$/.test(value)) return true;
  if (/^\d+[.)、]\s+/.test(value)) return false;
  if (/[。！？；.!?;,，、]$/.test(value)) return false;
  // A bare, short noun phrase on its own line is a common model output for a
  // subsection title. Do not mistake an unpunctuated operation sentence for a
  // title: those normally begin with an imperative/action verb instead.
  if (/^(?:按(?:下)?|请|将|若|如果|使用|清洁|检查|确认|打开|关闭|选择|调节|取出|放入|观察|确保|查看|更换|Remove|Check|Press|Please|Use|Clean|Install|Open|Close)/i.test(value)) return false;
  if (value.length <= 32 && /(?:模式|设置|步骤|方法|说明|状态|流程|要点|建议|原因|处理|方案|指南|概述|简介|操作|功能|规格|组件|部件|控制|显示|准备|存放|维护|清洁|安装|拆卸|更换|调节|排除)$/.test(value)) return true;
  return /(?:用途|作用|使用方法|注意事项|警告|安全|安装|拆卸|清洁|维护|更换|滤网|故障排除|部件|功能|规格|操作|准备|存放|搁架|烤架|餐具篮|battery|warning|safety|installation|removal|cleaning|maintenance|replacement|filter|troubleshooting|components?|features?|specifications?|operation|preparation|storage|usage|how to)/i.test(value)
    || /^[A-Z][A-Za-z0-9 /&'()_-]{2,48}$/.test(value);
}

function splitAnswerBlocks(text) {
  // Preserve common LLM output patterns: blank-line paragraphs, bullet lists,
  // numbered steps and bold section labels like **第一部分**.
  // Some providers flatten newlines while streaming. Restore a Markdown title
  // when it was appended after ordinary prose so the literal # does not leak
  // into the answer body.
  const normalized = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/([^\n])\s+(#{1,6})\s+(?=\S)/g, "$1\n\n$2 ");
  const lines = normalized.split("\n");
  const separated = [];
  let current = [];
  const flush = () => {
    if (current.some((line) => String(line || "").trim())) separated.push(current.join("\n"));
    current = [];
  };
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (isLikelyAnswerSubheading(trimmed, index < lines.length - 1)) {
      flush();
      separated.push(trimmed);
    } else {
      current.push(line);
    }
  });
  flush();
  return separated.join("\n\n")
    .replace(/\s+(\*\*(?:第[一二三四五六七八九十]+|首先|其次|再次|最后)[^*]{2,80}\*\*)/g, "\n\n$1")
    .split(/\n{2,}|\n(?=\s*(?:[-*]\s+|\d+[.、]\s+|\*\*))/)
    .map((block) => block.trim())
    .filter(Boolean);
}

function splitFlattenedMarkdownHeading(value) {
  const text = String(value || "").trim();
  // A flattened heading is normally followed by an operation sentence. Keep
  // only the title bold and restore the operation as ordinary paragraph text.
  const match = text.match(/^(.{2,48}?)\s+((?:检查|确认|按(?:下)?|请|将|若|如果|使用|清洁|拆(?:下|卸)|安装|选择|打开|关闭|调节|取出|放入|观察|确保|查看|更换|Remove|Check|Press|Please|Use|Clean|Install|Open|Close)[\s\S]*)$/i);
  return match
    ? { title: match[1].trim(), body: match[2].trim() }
    : { title: text, body: "" };
}

function createAnswerHeadingBlock(titleText, rest = "", level = 3) {
  const wrap = document.createElement("section");
  wrap.className = `answer-section answer-subheading-level-${Math.min(6, Math.max(1, level))}`;
  const title = document.createElement(level <= 2 ? "h3" : "h4");
  title.className = "answer-subheading-title";
  appendExplicitInlineMarkdown(title, String(titleText || "").trim().replace(/[：:。.]$/, ""));
  wrap.appendChild(title);
  if (String(rest || "").trim()) renderRichAnswerText(wrap, rest.trim());
  return wrap;
}

function createAnswerBlock(block) {
  // Minimal markdown-like renderer. We intentionally support a tiny subset
  // rather than arbitrary HTML/markdown so model output cannot inject markup.
  if (isProgramAnswerIntro(block)) {
    const intro = document.createElement("p");
    intro.className = "answer-intro";
    const strong = document.createElement("strong");
    strong.textContent = String(block || "").trim();
    intro.appendChild(strong);
    return intro;
  }
  // Fixed-answer data guarantees a manual heading is its own line.  Require
  // that boundary here as a final guard: unknown/model text such as
  // `# 程序显示 程序显示区…` remains ordinary text instead of making its body
  // a large bold heading.
  const markdownHeading = block.match(/^\s*(#{1,6})\s+([^\n]+)\n([\s\S]+)$/);
  if (markdownHeading) {
    return createAnswerHeadingBlock(
      markdownHeading[2].trim(),
      markdownHeading[3].trim(),
      markdownHeading[1].length,
    );
  }
  if (isLikelyAnswerSubheading(block, false)) {
    return createAnswerHeadingBlock(block, "", 3);
  }
  const titleMatch = block.match(/^\*\*((?:第[一二三四五六七八九十]+|首先|其次|再次|最后)[^*]{2,80})\*\*[：:。.]?\s*([\s\S]*)$/);
  if (titleMatch) {
    return createAnswerHeadingBlock(titleMatch[1], titleMatch[2], 3);
  }

  if (/^\s*[-*]\s+/.test(block)) {
    const ul = document.createElement("ul");
    ul.className = "answer-list";
    const items = block.split(/\n(?=\s*[-*]\s+)/).map((line) => line.replace(/^\s*[-*]\s+/, "").trim());
    for (const item of items.filter(Boolean)) {
      const li = document.createElement("li");
      appendInlineMarkdown(li, item);
      ul.appendChild(li);
    }
    return ul;
  }

  const p = document.createElement("p");
  appendInlineMarkdown(p, block);
  return p;
}

function appendInlineMarkdown(parent, text) {
  // Preserve line-level hierarchy even when a model omits Markdown around a
  // manual subsection label such as "制热模式 — 显示太阳图标".
  const lines = String(text || "").split("\n");
  lines.forEach((line, index) => {
    if (index > 0) parent.appendChild(document.createElement("br"));
    appendAnswerLine(parent, line);
  });
}

function structuredAnswerLabel(line) {
  const match = String(line || "").match(
    /^(\s*(?:（[^）\n]{1,24}）\s*)?[^：:—–\n]{2,36}?)(\s*(?:：|:|—|–)\s*.+)$/,
  );
  if (!match) return null;
  const prefix = match[1].trim();
  const semanticPrefix = prefix.replace(/^（[^）]+）\s*/, "");
  if (
    /^[\d一二三四五六七八九十]+[.、)]/.test(semanticPrefix)
    || /[。！？；]/.test(semanticPrefix)
    || !/(?:模式|功能|运行|方法|步骤|注意事项|准备工作|安装|拆卸|设置|说明)/.test(semanticPrefix)
  ) return null;
  return {
    prefix,
    suffix: match[2],
    level: /^(?:基本|高级|其他|可选)|按.+切换/.test(semanticPrefix) ? "group" : "item",
  };
}

function appendAnswerLine(parent, line) {
  if (isProgramAnswerIntro(line)) {
    const strong = document.createElement("strong");
    strong.className = "answer-intro-inline";
    strong.textContent = String(line || "").trim();
    parent.appendChild(strong);
    return;
  }
  const structured = !String(line || "").includes("**") ? structuredAnswerLabel(line) : null;
  if (structured) {
    const strong = document.createElement("strong");
    strong.className = `answer-structured-label ${structured.level}`;
    strong.textContent = structured.prefix;
    parent.appendChild(strong);
    appendExplicitInlineMarkdown(parent, structured.suffix);
    return;
  }
  appendExplicitInlineMarkdown(parent, line);
}

function isProgramAnswerIntro(value) {
  return /^\s*\u5173\u4e8e(?:[\u201c\"\u300c])?.{1,600}?(?:[\u201d\"\u300d])?\u7684\u95ee\u9898[\uff0c,]\u56de\u7b54\u5982\u4e0b[\uff1a:]\s*$/.test(String(value || ""));
}

function appendExplicitInlineMarkdown(parent, text) {
  // Support only `**bold**` emphasis. Everything else is appended as text nodes,
  // which keeps rendered model output safe without a sanitizer dependency.
  const parts = String(text || "").split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (!part) continue;
    const strongMatch = part.match(/^\*\*([^*]+)\*\*$/);
    if (strongMatch) {
      const strong = document.createElement("strong");
      strong.textContent = strongMatch[1].trim();
      parent.appendChild(strong);
    } else {
      parent.appendChild(document.createTextNode(part));
    }
  }
}

function getImageCaption(product, imageName) {
  const key = `${product || ""}|${imageName || ""}`;
  if (!imageCaptionCache.has(key)) {
    const url = new URL(IMAGE_CAPTION_ENDPOINT, window.location.origin);
    url.searchParams.set("product", product || "");
    url.searchParams.set("image", imageName || "");
    // Captions can be repaired independently from a completed chat response.
    // Do not reuse a previously cached empty response from the browser/CDN.
    url.searchParams.set("v", "caption-fix-20260731");
    const request = fetch(url, { cache: "no-store" })
      .then((response) => response.ok ? response.json() : null)
      .then((payload) => String(payload?.data?.caption || "").trim())
      .catch(() => "");
    // An empty response is transient while captions are rebuilt. Never let it
    // poison the current browser session and prevent a later repaired lookup.
    imageCaptionCache.set(key, request.then((caption) => {
      if (!caption) imageCaptionCache.delete(key);
      return caption;
    }));
  }
  return imageCaptionCache.get(key);
}

function captionTableRow(line) {
  const value = String(line || "").trim();
  if (!value.includes("|")) return null;
  const cells = value.replace(/^\|\s*/, "").replace(/\s*\|$/, "").split("|")
    .map((cell) => cell.trim());
  return cells.length >= 2 && cells.some(Boolean) ? cells : null;
}

function captionTableSeparator(line) {
  const cells = captionTableRow(line);
  return Boolean(cells && cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell)));
}

function appendCaptionContent(container, caption) {
  const lines = String(caption || "").replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    if (captionTableRow(lines[index]) && captionTableSeparator(lines[index + 1])) {
      const headers = captionTableRow(lines[index]);
      index += 2;
      const rows = [];
      while (index < lines.length && captionTableRow(lines[index]) && !captionTableSeparator(lines[index])) {
        rows.push(captionTableRow(lines[index]));
        index += 1;
      }
      const table = document.createElement("table");
      table.className = "manual-image-caption-table";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      headers.forEach((cell) => {
        const th = document.createElement("th");
        th.textContent = cell;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      rows.forEach((row) => {
        const tr = document.createElement("tr");
        headers.forEach((_, cellIndex) => {
          const td = document.createElement("td");
          td.textContent = row[cellIndex] || "";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      container.appendChild(table);
      continue;
    }
    const prose = [];
    while (index < lines.length && !(captionTableRow(lines[index]) && captionTableSeparator(lines[index + 1]))) {
      prose.push(lines[index]);
      index += 1;
    }
    const text = prose.join("\n").trim();
    if (text) {
      const paragraph = document.createElement("p");
      paragraph.textContent = text;
      container.appendChild(paragraph);
    }
  }
}

function createImageFigure(img, product, captionOverride, captionTranslation = "") {
  // Manual image URLs are served by `server.py` through `/manual-images/`.
  // `img.file` contains the real extension; `img.name` is the stable stem shown
  // in captions and answer lists.
  const figure = document.createElement("figure");
  figure.className = "manual-image";
  const image = document.createElement("img");
  image.src = `${state.data.imageBase}${encodeURIComponent(img.file)}`;
  image.alt = img.name;
  image.loading = "lazy";
  const cap = document.createElement("figcaption");
  cap.textContent = img.name;
  figure.append(image, cap);
  const captionPromise = captionOverride === undefined
    ? getImageCaption(product, img.name)
    : Promise.resolve(String(captionOverride || ""));
  captionPromise.then((caption) => {
    const visibleCaption = caption || `图片标注加载中：${img.name}`;
    const captionNode = document.createElement("div");
    captionNode.className = "manual-image-caption";
    appendCaptionContent(captionNode, visibleCaption);
    figure.appendChild(captionNode);
    if (!caption && captionOverride === undefined) {
      // A completed response can render before a caption index finishes its
      // independent refresh. Retry once and replace the temporary state in
      // place instead of requiring the user to submit the question again.
      window.setTimeout(async () => {
        const recovered = await getImageCaption(product, img.name);
        if (!recovered) return;
        captionNode.innerHTML = "";
        appendCaptionContent(captionNode, recovered);
      }, 1200);
    }
    if (captionTranslation) {
      const translationNode = document.createElement("p");
      translationNode.className = "manual-image-caption-translation";
      translationNode.textContent = captionTranslation;
      figure.appendChild(translationNode);
    }
  });
  return figure;
}

function createApiProgress(stages = null, minDurationMs = MIN_PROGRESS_MS) {
  // Progress is tied to the request lifecycle rather than a fixed 5-second
  // timer. While the request is pending, the bar creeps toward 92%; after the
  // request settles, `finish()` animates it to 100% and resolves. This prevents
  // a misleading "complete" bar while a slow RAG/LLM request is still running.
  let done = false;
  let finishing = false;
  let pct = 0;
  let stageIndex = -1;
  const createdAt = performance.now();
  let elapsedTimer = null;
  let resolveFinished;
  const finished = new Promise((resolve) => {
    resolveFinished = resolve;
  });
  const progressStages = stages || [
    { pct: 0, text: "正在封装独立问答请求..." },
    { pct: 12, text: "正在校验输入格式和请求标识..." },
    { pct: 24, text: "正在识别咨询意图与输入模态..." },
    { pct: 38, text: "正在全产品手册知识库召回候选证据..." },
    { pct: 52, text: "正在重排候选片段并筛选高相关证据..." },
    { pct: 66, text: "正在定位手册章节、来源与关联图片..." },
    { pct: 78, text: "正在基于证据生成客服回复..." },
    { pct: 87, text: "正在执行知识约束与答案完整性检查..." },
    { pct: 91, text: "正在整理答案、来源和配图..." },
  ];

  if (state.live) state.live.running = true;
  resetProgressTerminal();
  setProgressStage("提交中");
  setProgressElapsed(0);
  setProgressPercent(0);
  setProgressStatus(progressStages[0].text);
  appendProgressTerminal("system", "request created; waiting for backend progress events", 0);

  elapsedTimer = window.setInterval(() => {
    setProgressElapsed((performance.now() - createdAt) / 1000);
  }, 100);

  function tick() {
    // Slow asymptotic progress: early feedback feels responsive, then the bar
    // naturally slows down near 92% so it can wait for the real API.
    if (done) return;
    const remaining = 92 - pct;
    const step = Math.max(0.12, remaining * 0.018);
    pct = Math.min(92, pct + step);
    setProgressPercent(pct);
    const nextIndex = progressStages.findLastIndex((stage) => pct >= stage.pct);
    if (nextIndex !== stageIndex) {
      stageIndex = nextIndex;
      setProgressStage(progressStageName(progressStages[nextIndex].text));
      setProgressStatus(progressStages[nextIndex].text);
    }
    window.setTimeout(tick, 120);
  }
  tick();

  return {
    finish(finalText = "智能体客服已经生成回复", finalKind = "api") {
      // Idempotent finish: if two error/success paths call it, the original
      // promise is reused and the animation is not duplicated.
      if (done || finishing) return finished;
      finishing = true;
      const elapsed = performance.now() - createdAt;
      const holdMs = Math.max(0, minDurationMs - elapsed);
      setProgressStatus(holdMs > 0 ? "接口已返回，正在整理客服回复..." : "接口已返回，正在渲染客服回复...");
      window.setTimeout(() => {
        done = true;
        const start = pct;
        const startTime = performance.now();
        function frame(now) {
          const local = Math.min(1, (now - startTime) / 420);
          const eased = 1 - Math.pow(1 - local, 3);
          const nextPct = start + (100 - start) * eased;
          setProgressPercent(nextPct);
          if (local < 1) {
            requestAnimationFrame(frame);
            return;
          }
          window.setTimeout(() => {
            if (elapsedTimer) {
              window.clearInterval(elapsedTimer);
              elapsedTimer = null;
            }
            setProgressElapsed((performance.now() - createdAt) / 1000);
            if (state.live) state.live.running = false;
            setProgressStage(finalKind === "error" ? "失败" : "完成");
            setProgressPercent(100);
            setProgressStatus(finalText, finalKind);
            appendProgressTerminal(finalKind === "error" ? "error" : "done", finalText, (performance.now() - createdAt) / 1000);
            resolveFinished();
          }, 180);
        }
        requestAnimationFrame(frame);
      }, holdMs);
      return finished;
    },
    };
  }

function startProgressLogPolling(requestId) {
  let stopped = false;
  const seen = new Set();
  async function poll() {
    if (stopped || !requestId) return;
    try {
      const res = await fetch(`${PROGRESS_ENDPOINT}?request_id=${encodeURIComponent(requestId)}`, {
        cache: "no-store",
      });
      if (res.ok) {
        const payload = await res.json();
        const events = payload?.data?.events || [];
        if (events.length) {
          const latest = events[events.length - 1];
          setProgressStage(progressStageLabel(latest.stage));
          setProgressStatus(String(latest.message || "").trim());
          for (const event of events) {
            const key = `${event.elapsed}:${event.stage}:${event.message}`;
            if (seen.has(key)) continue;
            seen.add(key);
            appendProgressTerminal(event.stage, event.message, event.elapsed);
          }
        }
      }
    } catch (_error) {
      // Progress polling is auxiliary; the main /chat request owns success/failure.
    }
    if (!stopped) window.setTimeout(poll, 700);
  }
  poll();
  return {
    stop() {
      stopped = true;
    },
  };
}


function formatProgressEvents(events) {
  return events
    .map((event) => {
      const elapsed = Number.isFinite(event.elapsed) ? `${event.elapsed.toFixed(1)}s` : "--";
      return `[${elapsed}] ${event.message || ""}`;
    })
    .join("\n");
}

// ===== Per-question RAG process records =====
// A "process record" captures one question's whole RAG flow (stage, status,
// progress, and the full event log). The right sidebar renders exactly one
// record at a time. Live updates write into `state.live`; the sidebar only
// repaints when the live record is also the active (currently-viewed) one.

// Turn the backend's terse / bilingual progress messages into plain, friendly
// Chinese so the right-sidebar "thinking process" reads like a human narration
// of what the RAG pipeline is doing. Pure presentation: the raw events are kept
// on the record; this only changes how they are shown.
const PROGRESS_RULES = [
  [/request created.*$/i, "建立请求，准备检索手册"],
  [/input accepted:\s*(\d+)\s*text characters,\s*(\d+)\s*image attachments/i, "输入解析完成：文本 $1 字符，图片附件 $2 张"],
  [/conversation policy:\s*stateless independent question;\s*history context disabled/i, "会话策略：本轮独立问答，不注入历史上下文"],
  [/conversation policy:\s*stateless independent question;\s*history context requested/i, "会话策略：收到历史上下文请求"],
  [/retrieval scope:\s*full product-manual knowledge base;\s*quick-product selection is not a retrieval filter/i, "检索范围：全部产品手册；快捷产品选择不限制 RAG"],
  [/generation profile:\s*([^;]+);\s*reasoning effort\s*(.+)$/i, "生成配置：$1，推理强度 $2"],
  [/connected to independent retrieval service;\s*streaming progress channel established/i, "已连接独立检索服务，并建立实时进度通道"],
  [/routing complete:\s*customer-service response/i, "路由完成：进入通用客服回答模式"],
  [/routing complete:\s*manual knowledge response/i, "路由完成：进入产品手册知识回答模式"],
  [/evidence summary:\s*(\d+)\s*source chunks across\s*(\d+)\s*manuals(?:\s*\((.*)\))?/i, "证据汇总：采用 $1 个知识片段，来自 $2 本手册 $3"],
  [/reranking summary:\s*(\d+)\s*scored candidates;\s*top relevance\s*([\d.]+)/i, "重排结果：$1 个候选完成评分，最高相关度 $2"],
  [/image evidence:\s*(\d+)\s*figures selected and bound to the answer/i, "图片证据：筛选并绑定 $1 张相关手册图片"],
  [/grounding policy applied:.*no prior-turn context injected/i, "知识约束：回答依据检索证据生成，未注入上一轮内容"],
  [/enter compact RAG.*$/i, "开始检索产品手册证据"],
  [/using fast verified reference evidence.*$/i, "命中已验证的手册证据"],
  [/检索完成：候选证据\s*(\d+)[^]*?命中手册\s*(\d+)[^]*/, "检索到 $1 条候选证据，涉及 $2 本手册"],
  [/compact evidence selected.*$/i, "已挑选最相关的证据片段"],
  [/evidence compressed[^]*calling model.*$/i, "压缩证据，调用大模型生成回答"],
  [/model returned[^]*parsing.*$/i, "大模型已返回，正在解析答案"],
  [/模型答案为空[^]*兜底.*$/i, "模型未输出，改用已验证手册原文"],
  [/模型答案为空[^]*$/i, "模型未输出，准备降级处理"],
  [/开始根据[^]*图片[^]*$/i, "开始校验与补齐配图"],
  [/构建图片上下文.*$/i, "分析图片所在的章节上下文"],
  [/校正同块图片顺序.*$/i, "校正图片为手册原始顺序"],
  [/重定位尾部图片.*$/i, "把图片移到对应的步骤旁"],
  [/补齐\s*compact RAG\s*任务图片.*$/i, "补齐该问题应有的图片"],
  [/裁剪弱相关结构图.*$/i, "去除不相关的图片"],
  [/执行图片预算.*$/i, "控制图片数量在合理范围"],
  [/绑定裸\s*PIC\s*标记.*$/i, "把图片绑定到正文位置"],
  [/清理代码包裹图片标记.*$/i, "整理图片标记格式"],
  [/reference chunk 缺图补齐.*$/i, "补齐参考证据中缺失的图片"],
  [/单\s*reference chunk 图片恢复.*$/i, "恢复参考证据的全部图片"],
  [/健身追踪器答案规范化.*$/i, "规范化答案格式"],
  [/发电机热机安全图补齐.*$/i, "补齐安全提示相关图片"],
  [/生成提交格式.*$/i, "生成最终答案与配图"],
  [/证据图片范围过滤完成：输出\s*(\d+)[^]*/, "完成配图：最终输出 $1 张图"],
  [/API 调用成功.*$/i, "回答生成完成"],
  [/接口调用失败.*$/i, "调用失败，请重试"],
  [/跳过弱相关结构图裁剪.*$/i, "保留全部证据图片"],
];

function humanizeProgress(message, stage = "") {
  let t = String(message || "").trim();
  if (!t) return "";
  const replacementCount = (t.match(/\uFFFD/g) || []).length;
  if (replacementCount >= 2) {
    const fallbacks = {
      accepted: "检索服务已接收本次请求",
      route: "正在识别咨询类型并定位产品领域",
      start: "正在初始化检索与证据处理流程",
      retrieve: "正在召回并筛选产品手册证据",
      model: "正在执行模型处理阶段",
      postprocess: "正在整理答案结构与引用关系",
      images: "正在匹配并校验相关手册图片",
      done: "当前处理阶段已完成",
    };
    return fallbacks[String(stage || "").toLowerCase()] || "正在执行当前处理阶段";
  }
  t = t.replace(/^图片后处理步骤完成：/, "");
  // Match friendly rules first (they own the whole message)...
  for (const [re, rep] of PROGRESS_RULES) {
    if (re.test(t)) return t.replace(re, rep).trim();
  }
  // ...then, only as a fallback, strip trailing timing noise.
  t = t.replace(/[，,]?\s*(?:用时|耗时)\s*[\d.]+\s*s.*$/i, "");
  t = t.replace(/[;；]?\s*\bin\s+[\d.]+s\b.*$/i, "");
  return t.trim();
}

const IDLE_PROCESS = { id: "__idle__", stage: "待命", status: "等待提问", elapsed: 0, percent: 0, running: false, kind: null, events: [], images: [], summary: null };

function startProcess({ question = "", requestKind = null } = {}) {
  // Create a fresh record for a new question and make it the active view.
  const id = `proc_${++state.procSeq}`;
  const rec = {
    id,
    stage: "提交中",
    status: "",
    elapsed: 0,
    percent: 0,
    running: true,
    kind: null,
    requestKind,
    question,
    events: [],
    images: [],
    summary: null,
    audit: null,
  };
  state.processes.set(id, rec);
  state.live = rec;
  state.activeProcessId = id;
  renderProcess(rec);
  // On phones the audit timeline lives in a drawer; open it with a request so
  // the live retrieval and generation stages are not hidden behind a control.
  if (window.matchMedia("(max-width: 820px)").matches && els.appShell) {
    openMobileLogbar();
  }
  return rec;
}

function auditNumber(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function auditRank(value) {
  return value === null || value === undefined || value === "" ? "—" : `#${escapeHtml(String(value))}`;
}

function hasUsableAuditTrace(trace) {
  return Boolean(window.RagAuditContract?.hasMeaningfulAuditTrace?.(trace));
}

function mergeAuditTrace(previous, incoming) {
  if (!window.RagAuditContract?.mergeAuditTrace) return incoming || previous || null;
  return window.RagAuditContract.mergeAuditTrace(previous, incoming);
}

function auditViewKind(trace, rec) {
  return window.RagAuditContract?.auditViewKind?.(trace, rec) || "manual_rag";
}

function renderAuditTrace(trace) {
  if (!trace || typeof trace !== "object") return "";
  const route = trace.route || {};
  const query = trace.query || {};
  const retrieval = trace.retrieval || {};
  const evidence = trace.evidence || {};
  const timings = trace.timings || {};
  const candidates = Array.isArray(retrieval.candidates) ? retrieval.candidates : [];
  const selected = Array.isArray(evidence.selected) ? evidence.selected : [];
  const routeCandidates = Array.isArray(route.candidates) ? route.candidates : [];
  const safeText = (value, limit = 320) => escapeHtml(compactText(value, limit));
  const candidateRows = candidates.map((candidate) => {
    const role = candidate.selected
      ? (candidate.final_rank === 1 || candidate.evidence_role === "primary" ? "主证据" : "辅助证据")
      : "未入选";
    const rerank = [candidate.keyword_rerank_rank, candidate.original_rerank_rank]
      .filter((value) => value !== null && value !== undefined)
      .map((value) => `#${value}`).join(" / ") || "—";
    return `<tr class="${candidate.selected ? "audit-selected" : ""}">
      <td>${auditRank(candidate.rrf_rank)}</td>
      <td><b>${escapeHtml(String(candidate.chunk_id ?? "—"))}</b><small>${safeText(candidate.heading, 72)}</small></td>
      <td>${auditNumber(candidate.bm25_raw, 3)}<small>相对 ${auditNumber(candidate.bm25_relative, 3)}</small></td>
      <td>${auditNumber(candidate.dense_cosine, 3)}</td>
      <td>${auditNumber(candidate.rrf_score, 5)}</td>
      <td>${escapeHtml(rerank)}</td>
      <td>${escapeHtml(role)}<small>${candidate.final_rank ? `最终 #${candidate.final_rank}` : ""}</small></td>
    </tr>`;
  }).join("");
  const evidenceRows = selected.map((item) => `<article class="audit-evidence">
    <div><span class="audit-role ${item.role === "primary" ? "primary" : "supporting"}">${item.role === "primary" ? "主 Chunk" : "辅助 Chunk"}</span><b>${escapeHtml(String(item.chunk_id || "—"))}</b></div>
    <strong>${safeText(item.heading, 180)}</strong>
    <p>${safeText(item.excerpt, 460)}</p>
  </article>`).join("");
  return `<section class="audit-trace" aria-label="详细检索审计">
    <div class="audit-title">检索审计详情 <span>真实链路数据</span></div>
    <div class="audit-grid">
      <div><label>锁定手册</label><b>${safeText(route.selected_manual || "未确定", 90)}</b><small>${safeText(route.reason || "", 110)} · ${safeText(route.confidence || "", 28)}</small></div>
      <div><label>手册候选</label><b>${safeText(routeCandidates.join("、") || "—", 130)}</b><small>${route.candidate_scores?.length ? route.candidate_scores.map((item) => `${safeText(item.manual, 30)} ${auditNumber(item.score, 3)}`).join(" · ") : ""}</small></div>
      <div class="audit-wide"><label>原始问题</label><b>${safeText(query.original || "—", 380)}</b></div>
      <div><label>BM25 Query</label><b>${safeText(query.sparse || "—", 180)}</b></div>
      <div><label>Dense Query</label><b>${safeText(query.semantic || "—", 180)}</b></div>
    </div>
    <div class="audit-timings"><span>检索 ${auditNumber(timings.retrieval_seconds)}s</span><span>生成 ${auditNumber(timings.generation_seconds)}s</span><span>其他 ${auditNumber(timings.other_seconds)}s</span><b>总计 ${auditNumber(timings.total_seconds)}s</b></div>
    <details class="audit-details" open>
      <summary>候选召回与排序（${candidates.length} 条；已过滤 ${Number(retrieval.filtered_count || 0)} 条）</summary>
      <div class="audit-table-wrap"><table class="audit-table"><thead><tr><th>RRF</th><th>Chunk / 标题</th><th>BM25</th><th>Dense</th><th>RRF 分</th><th>Rerank</th><th>最终角色</th></tr></thead><tbody>${candidateRows || '<tr><td colspan="7">本次没有可展示的候选数据。</td></tr>'}</tbody></table></div>
    </details>
    <details class="audit-details" open>
      <summary>最终送入生成模型的证据（${selected.length} 节，${Number(evidence.context_chars || 0)} 字符）</summary>
      <div class="audit-evidence-list">${evidenceRows || "未返回证据范围。"}</div>
    </details>
  </section>`;
}

function routeReasonLabel(reason) {
  const value = String(reason || "").toLowerCase();
  if (value.includes("request_product")) return "根据问题中的产品名称锁定";
  if (value.includes("content_vote")) return "根据手册内容匹配锁定";
  if (value.includes("explicit")) return "根据明确产品信息锁定";
  return "根据当前问题与手册内容锁定";
}

function renderKeyProcessTimeline(rec) {
  const trace = rec?.audit;
  const hasTrace = trace && typeof trace === "object";
  const route = hasTrace ? (trace.route || {}) : {};
  const query = hasTrace ? (trace.query || {}) : {};
  const retrieval = hasTrace ? (trace.retrieval || {}) : {};
  const timings = hasTrace ? (trace.timings || {}) : {};
  const candidates = Array.isArray(retrieval.candidates) ? retrieval.candidates.slice(0, 8) : [];
  const raw = Array.isArray(rec?.events) ? rec.events : [];
  const visibleStages = new Set(["accepted", "start", "analyze", "input", "vision", "images", "route", "classify", "classification", "scope", "knowledge", "retrieve", "search", "tool", "compose", "model", "generate", "finalize", "done", "complete", "error"]);
  const stageAliases = {
    accepted: "input", start: "input", analyze: "input", input: "input",
    vision: "vision", images: "images",
    route: "scope", classify: "scope", classification: "scope", scope: "scope",
    knowledge: "retrieve", retrieve: "retrieve", search: "retrieve", tool: "retrieve",
    compose: "model", model: "model", generate: "model", finalize: "model",
    done: "done", complete: "done", error: "error",
  };
  const items = [];
  for (const ev of raw) {
    const rawStage = String(ev.stage || "").toLowerCase();
    if (!visibleStages.has(rawStage)) continue;
    const stage = stageAliases[rawStage] || rawStage;
    const msg = humanizeProgress(ev.message, ev.stage);
    if (!msg) continue;
    const existing = items.find((item) => item.stage === stage);
    if (existing) {
      existing.msg = msg;
      existing.elapsed = ev.elapsed;
    } else {
      items.push({ stage, msg, elapsed: ev.elapsed });
    }
  }

  // A completed trace enriches the same live timeline instead of replacing it.
  // Missing stages are inserted in process order so old records remain auditable.
  const stageOrder = ["input", "vision", "scope", "retrieve", "model", "images", "done", "error"];
  const traceStages = hasTrace ? ["input", "scope", "retrieve", "model"] : [];
  for (const stage of traceStages) {
    if (!items.some((item) => item.stage === stage)) items.push({ stage, msg: "", elapsed: null });
  }
  items.sort((left, right) => stageOrder.indexOf(left.stage) - stageOrder.indexOf(right.stage));
  if (!items.length) return "";

  const stageTitle = {
    input: "问题解析",
    vision: "读图与图片理解",
    images: "配图校验",
    scope: "手册锁定",
    retrieve: "混合召回",
    model: "答案生成",
    done: "处理完成",
    error: "处理失败",
  };
  const rankCards = candidates.map((candidate) => {
    const selectedRole = candidate.selected
      ? (candidate.final_rank === 1 || candidate.evidence_role === "primary" ? "主证据" : "辅助")
      : "候选";
    const rerank = [candidate.keyword_rerank_rank, candidate.original_rerank_rank]
      .filter((value) => value !== null && value !== undefined)
      .map((value) => `#${value}`).join("/") || "—";
    return `<div class="process-rank-card ${candidate.selected ? "is-selected" : ""}">
      <div><b>#${escapeHtml(String(candidate.rrf_rank || "—"))} · ${escapeHtml(String(candidate.chunk_id || "—"))}</b><span class="process-rank-role ${candidate.selected ? "selected" : ""}">${selectedRole}</span></div>
      <p>${escapeHtml(compactText(candidate.heading || "未命名片段", 100))}</p>
      <small>BM25 ${auditNumber(candidate.bm25_raw)} · Dense ${auditNumber(candidate.dense_cosine)} · RRF ${auditNumber(candidate.rrf_score, 5)} · 重排 ${rerank}</small>
    </div>`;
  }).join("");
  const detailFor = (stage) => {
    if (!hasTrace) return "";
    if (stage === "input") return `<div class="tl-detail-grid">
      <div class="tl-detail-wide"><label>原始问题</label><span>${escapeHtml(compactText(query.original || "—", 260))}</span></div>
      <div><label>BM25 Query</label><span>${escapeHtml(compactText(query.sparse || "—", 180))}</span></div>
      <div><label>Dense Query</label><span>${escapeHtml(compactText(query.semantic || "—", 180))}</span></div>
    </div>`;
    if (stage === "scope") {
      const routeCandidates = Array.isArray(route.candidates) ? route.candidates.join("、") : "";
      return `<div class="tl-detail-grid"><div><label>锁定手册</label><span>${escapeHtml(compactText(route.selected_manual || "未确定", 110))}</span></div><div><label>锁定依据</label><span>${escapeHtml(routeReasonLabel(route.reason))}</span></div>${routeCandidates ? `<div class="tl-detail-wide"><label>候选手册</label><span>${escapeHtml(compactText(routeCandidates, 180))}</span></div>` : ""}</div>`;
    }
    if (stage === "retrieve") return `<div class="tl-detail-summary">BM25 + Dense + RRF · ${Number(retrieval.candidate_count || candidates.length)} 个候选 · 过滤 ${Number(retrieval.filtered_count || 0)} 个</div><div class="process-rank-list">${rankCards || "未返回候选排名。"}</div>`;
    if (stage === "model") return `<div class="audit-timings"><span>检索 ${auditNumber(timings.retrieval_seconds)}s</span><span>生成 ${auditNumber(timings.generation_seconds)}s</span><b>总计 ${auditNumber(timings.total_seconds)}s</b></div>`;
    return "";
  };

  return items.map((item, index) => {
    const isLast = index === items.length - 1;
    const stateClass = item.stage === "error" ? "err" : (rec.running && isLast ? "active" : "done");
    const title = stageTitle[item.stage] || progressStageLabel(item.stage);
    const message = item.msg && item.msg !== title ? item.msg : "";
    const sec = Number.isFinite(Number(item.elapsed)) ? `${Number(item.elapsed).toFixed(1)}s` : "";
    return `<div class="tl-item tl-${stateClass} stage-${escapeHtml(item.stage)}">
      <span class="tl-dot" aria-hidden="true"></span>
      <div class="tl-body">
        <div class="tl-heading"><span class="tl-seq">${String(index + 1).padStart(2, "0")}</span><b>${escapeHtml(title)}</b>${sec ? `<time>${sec}</time>` : ""}</div>
        ${message ? `<div class="tl-msg">${escapeHtml(message)}</div>` : ""}
        ${detailFor(item.stage)}
      </div>
    </div>`;
  }).join("");
}

const AUDIT_VIEW_META = {
  manual_rag: {
    label: "普通手册召回",
    description: "手册路由、混合检索、证据取舍与答案生成",
  },
  visual_manual: {
    label: "图片 / 链接手册流程",
    description: "媒体解析、视觉识别、图片向量匹配与手册处理",
  },
  customer_service: {
    label: "纯客服题",
    description: "本地意图判断、历史上下文与客服回答生成",
  },
};

function auditValue(value, limit = 360) {
  if (value === true) return "是";
  if (value === false) return "否";
  if (Array.isArray(value)) return compactText(value.join("、"), limit);
  if (value && typeof value === "object") return compactText(JSON.stringify(value), limit);
  return compactText(value ?? "", limit);
}

function auditHasValue(value) {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function renderAuditRows(rows) {
  const visible = rows.filter((row) => auditHasValue(row.value));
  if (!visible.length) return "";
  return `<div class="audit-kv-list">${visible.map((row) => `<div class="audit-kv-row${row.wide ? " audit-kv-wide" : ""}">
    <label>${escapeHtml(row.label)}</label>
    <span class="${row.mono ? "audit-mono" : ""}">${escapeHtml(auditValue(row.value, row.limit || 420))}</span>
  </div>`).join("")}</div>`;
}

function auditRouteReason(reason) {
  const labels = {
    verified_manual_image_or_user_product: "由已验证图片或用户明确产品锁定",
    explicit_product_title_candidates: "由问题中的明确产品标题锁定",
    visual_plus_caption_plus_text_candidates: "综合视觉、图片说明与问题文字锁定",
    cross_manual_caption_candidates: "跨手册图片说明参与候选比较",
    full_corpus: "问题未锁定单一手册，执行全库检索",
  };
  return labels[String(reason || "")] || routeReasonLabel(reason);
}

function auditDropReason(reason) {
  const labels = {
    low_rrf_optional_over_budget: "证据包过大，按较低 RRF / 重排效用剔除",
    auxiliary_evidence_disabled: "辅助证据当前被禁用",
  };
  return labels[String(reason || "")] || String(reason || "未记录原因");
}

function collectAuditHits(trace) {
  const hits = [];
  for (const event of Array.isArray(trace?.events) ? trace.events : []) {
    for (const hit of Array.isArray(event?.retrieval_hits) ? event.retrieval_hits : []) hits.push(hit);
  }
  return hits;
}

function isEvidenceReplay(trace) {
  return String(trace?.mode || "").toLowerCase().includes("evidence_replay");
}

function renderCandidateRanking(trace) {
  const retrieval = trace?.retrieval || {};
  const candidates = Array.isArray(retrieval.candidates) ? retrieval.candidates : [];
  if (!candidates.length) return `<div class="audit-inline-note">召回排名尚未返回。</div>`;
  const selectedEvidence = Array.isArray(trace?.evidence?.selected) ? trace.evidence.selected : [];
  const selectedById = new Map(selectedEvidence.map((item) => [String(item.chunk_id), item]));

  return `<div class="audit-rank-list">${candidates.map((candidate, index) => {
    const chunkId = String(candidate.chunk_id ?? candidate.matched_chunk_id ?? "—");
    const selectedItem = selectedById.get(chunkId);
    const selected = Boolean(selectedItem || candidate.selected);
    const rawRole = String(selectedItem?.tier || candidate.evidence_role || "").toLowerCase();
    const role = !selected
      ? (isEvidenceReplay(trace) ? "未选为检索证据" : "未送入模型")
      : (rawRole === "core" || rawRole === "primary" ? "主证据" : "辅助证据");
    const rerank = [candidate.keyword_rerank_rank, candidate.original_rerank_rank]
      .filter((value) => value !== null && value !== undefined)
      .map((value) => `#${value}`).join(" / ") || "—";
    const channelRanks = Object.entries(candidate.channel_ranks || {})
      .map(([name, rank]) => `${name} #${rank}`).join(" · ");
    return `<div class="audit-rank-row${selected ? " is-selected" : ""}">
      <div class="audit-rank-heading">
        <b><span>${String(index + 1).padStart(2, "0")}</span> Chunk ${escapeHtml(chunkId)}</b>
        <em class="audit-evidence-role${role === "主证据" ? " is-primary" : ""}">${escapeHtml(role)}</em>
      </div>
      <p>${escapeHtml(compactText(candidate.heading || "未命名片段", 150))}</p>
      <div class="audit-score-line">
        <span>RRF 排名 <b>${auditRank(candidate.rrf_rank)}</b></span>
        <span>BM25 <b>${auditNumber(candidate.bm25_raw, 3)}</b></span>
        <span>相对分 <b>${auditNumber(candidate.bm25_relative, 3)}</b></span>
        <span>Dense <b>${auditNumber(candidate.dense_cosine, 3)}</b></span>
        <span>RRF 分 <b>${auditNumber(candidate.rrf_score, 5)}</b></span>
        <span>重排 <b>${escapeHtml(rerank)}</b></span>
        ${candidate.final_evidence_rank || candidate.final_rank ? `<span>证据顺序 <b>#${escapeHtml(String(candidate.final_evidence_rank || candidate.final_rank))}</b></span>` : ""}
      </div>
      ${channelRanks ? `<small class="audit-channel-ranks">通道排名：${escapeHtml(channelRanks)}</small>` : ""}
    </div>`;
  }).join("")}</div>`;
}

function renderEvidenceSelection(trace) {
  const evidence = trace?.evidence || {};
  const selected = Array.isArray(evidence.selected) ? evidence.selected : [];
  const candidates = Array.isArray(trace?.retrieval?.candidates) ? trace.retrieval.candidates : [];
  const candidateById = new Map(candidates.map((item) => [String(item.chunk_id), item]));
  const hitById = new Map(collectAuditHits(trace).map((item) => [
    String(item.matched_chunk_id ?? item.chunk_id),
    item,
  ]));
  const budget = evidence.budget || trace?.generation_evidence_budget || {};
  const dropped = Array.isArray(budget.dropped) ? budget.dropped : [];
  const replay = isEvidenceReplay(trace);
  let output = "";

  if (selected.length) {
    output += `<div class="audit-evidence-list-flat">${selected.map((item) => {
      const chunkId = String(item.chunk_id ?? "—");
      const candidate = candidateById.get(chunkId) || {};
      const hit = hitById.get(chunkId) || {};
      const tier = String(item.tier || candidate.evidence_role || "related").toLowerCase();
      const role = tier === "core" || tier === "primary" ? "主证据" : "辅助证据";
      const excerpt = hit.matched_content || hit.content || item.excerpt || "";
      return `<div class="audit-evidence-row${role === "主证据" ? " is-primary" : ""}">
        <div><b>${escapeHtml(role)} · Chunk ${escapeHtml(chunkId)}</b><span>${escapeHtml(compactText(candidate.heading || hit.heading || "", 150))}</span></div>
        ${excerpt ? `<p>${escapeHtml(compactText(excerpt, 320))}</p>` : ""}
      </div>`;
    }).join("")}</div>`;
  } else {
    output += `<div class="audit-inline-note">证据清单正在整理；已返回的候选排名会保持显示。</div>`;
  }

  output += renderAuditRows([
    { label: replay ? "核对字符" : "送入字符", value: evidence.context_chars || budget.after_chars },
    { label: "压缩前", value: auditHasValue(budget.before_chars) ? `${budget.before_chunks ?? "—"} 块 / ${budget.before_chars} 字符` : "" },
    { label: "压缩后", value: auditHasValue(budget.after_chars) ? `${budget.after_chunks ?? "—"} 块 / ${budget.after_chars} 字符` : "" },
    { label: "预算上限", value: auditHasValue(budget.max_chars) ? `${budget.max_chunks ?? "—"} 块 / ${budget.max_chars} 字符` : "" },
    { label: "是否压缩", value: auditHasValue(budget.applied) ? budget.applied : "" },
  ]);
  if (dropped.length) {
    output += `<details class="audit-flat-details"><summary>${replay ? "未选为检索证据" : "未送入模型"}的片段（${dropped.length}）</summary><div class="audit-dropped-list">${dropped.map((item) => {
      const row = typeof item === "object" ? item : { chunk_id: item };
      return `<div><b>Chunk ${escapeHtml(String(row.chunk_id || "—"))}</b><span>${escapeHtml(auditDropReason(row.reason))}</span>${row.rrf_rank ? `<small>RRF #${escapeHtml(String(row.rrf_rank))}</small>` : ""}</div>`;
    }).join("")}</div></details>`;
  }
  return output;
}

function renderAnswerEvidenceAlignment(trace) {
  const alignment = trace?.answer_evidence_alignment || {};
  const chunks = Array.isArray(alignment.matched_chunks) ? alignment.matched_chunks : [];
  const blockCoverage = alignment.answer_block_coverage || {};
  const pictureCoverage = alignment.picture_coverage || {};
  let output = renderAuditRows([
    { label: "核对方法", value: alignment.method === "literal_overlap_and_picture_anchor" ? "原文重合 + 图片锚点" : alignment.method },
    { label: "对应片段", value: alignment.matched_chunk_count ?? chunks.length },
    { label: "正文覆盖", value: auditHasValue(blockCoverage.total) ? `${blockCoverage.matched || 0}/${blockCoverage.total} 个答案段落` : "" },
    { label: "图片覆盖", value: Array.isArray(pictureCoverage.required) ? `${(pictureCoverage.matched || []).length}/${pictureCoverage.required.length}` : "" },
    { label: "缺失图片", value: pictureCoverage.missing },
  ]);
  if (chunks.length) {
    output += `<div class="audit-evidence-list-flat">${chunks.map((chunk) => `<div class="audit-evidence-row is-primary">
      <div><b>原文对应 · Chunk ${escapeHtml(String(chunk.chunk_id || "—"))}</b><span>${escapeHtml(compactText(chunk.heading || "", 150))}</span></div>
      <p>${escapeHtml((chunk.match_reasons || []).map((reason) => String(reason).startsWith("picture:") ? `图片 ${String(reason).slice(8)}` : "文字内容命中").join(" · "))}${auditHasValue(chunk.alignment_score) ? ` · 对齐分 ${auditNumber(chunk.alignment_score, 3)}` : ""}</p>
    </div>`).join("")}</div>`;
  } else {
    output += `<div class="audit-inline-note">尚未找到足以解释最终答案的原文片段；检索排名仍按真实结果展示。</div>`;
  }
  return output;
}

function renderVisualVector(trace) {
  const visual = trace?.visual_preroute || {};
  const vector = visual.vector_trace || {};
  const hits = Array.isArray(vector.hits) ? vector.hits : [];
  const visibleHits = hits.slice(0, 10);
  let output = renderAuditRows([
    { label: "匹配引擎", value: vector.index ? "本地 DINOv2" : (visual.strategy || visual.provider) },
    { label: "检索范围", value: vector.scope },
    { label: "向量候选", value: visual.vector_candidate },
    { label: "最高分", value: auditHasValue(vector.top_score) ? vector.top_score : visual.vector_score, mono: true },
    { label: "是否采纳", value: auditHasValue(vector.accepted) ? vector.accepted : "" },
    { label: "决策", value: visual.decision || vector.reason },
    { label: "耗时", value: auditHasValue(vector.elapsed_s) ? `${vector.elapsed_s}s` : "", mono: true },
  ]);
  if (visibleHits.length) {
    output += `<div class="audit-stage-summary">显示前 ${visibleHits.length} 个向量候选，共 ${hits.length} 个</div><div class="audit-vector-hits">${visibleHits.map((hit, index) => `<div>
      <b>#${index + 1} · ${escapeHtml(String(hit.image_id || "未命名图片"))}</b>
      <span>${escapeHtml(compactText(hit.product || "", 90))}</span>
      <small>相似度 ${auditNumber(hit.visual_score, 5)}${hit.caption ? ` · ${escapeHtml(compactText(hit.caption, 130))}` : ""}</small>
    </div>`).join("")}</div>`;
  }
  return output || `<div class="audit-inline-note">本次未产生可展示的图片向量候选。</div>`;
}

function renderThreeWayImageRetrieval(trace, hasInputImage = false) {
  const visual = trace?.visual_preroute || {};
  const match = visual.manual_image_match || {};
  const candidates = Array.isArray(match.retrieval_candidates) ? match.retrieval_candidates : [];
  if (!candidates.length && !match.caption_description) {
    if (!hasInputImage) return "";
    const reason = visual.manual_grounding
      || (visual.vector_manual_image_match ? "已由本地图像向量直接锚定" : "本轮没有形成可复核的手册图片候选");
    return `${renderAuditRows([
      { label: "执行状态", value: "本轮未进入 Caption BM25 + Dense + DINOv2 + Qwen-VL 原图复核" },
      { label: "原因", value: reason, wide: true },
      { label: "当前图片命中", value: match.image_id || visual.vector_manual_image_match?.image_id },
      { label: "当前产品", value: visual.product },
    ])}<div class="audit-inline-note">此阶段始终保留，避免缓存、直接向量锚定或候选不足时让图片召回明细看起来消失。</div>`;
  }
  const selection = match.selection || {};
  const selectedId = selection.image_id || match.image_id;
  const timings = match.timings || {};
  let output = renderAuditRows([
    { label: "检索策略", value: "图片描述 + Caption BM25 + Caption Dense + DINOv2，经 RRF 融合后由 Qwen-VL 对候选原图复核", wide: true },
    { label: "图片描述", value: match.caption_description, wide: true },
    { label: "检索词", value: match.caption_search_terms, wide: true },
    { label: "融合候选", value: candidates.length ? `${candidates.length} 张手册图` : "" },
    { label: "最终命中", value: selectedId },
    { label: "复核置信度", value: selection.confidence || match.confidence },
    { label: "可核验理由", value: selection.reason || match.reason, wide: true },
    { label: "命中章节", value: match.heading, wide: true },
  ]);
  if (candidates.length) {
    output += `<div class="audit-stage-summary">RRF 排名由三条独立召回通道共同决定；数字为该图片在各通道内的名次，空白表示未进入 DINOv2 候选。</div>`;
    output += `<div class="image-rrf-list">${candidates.map((candidate) => {
      const selected = String(candidate.image_id || "") === String(selectedId || "");
      return `<div class="image-rrf-row${selected ? " is-selected" : ""}">
        <div><b>#${escapeHtml(String(candidate.rrf_rank || "-"))} · ${escapeHtml(String(candidate.image_id || "未命名图片"))}</b>${selected ? `<span>已选中</span>` : ""}</div>
        <small>${escapeHtml(compactText(candidate.product || "", 60))} · BM25 #${escapeHtml(String(candidate.bm25_rank || "-"))} · Dense #${escapeHtml(String(candidate.dense_rank || "-"))} · DINOv2 #${escapeHtml(String(candidate.image_rank || "-"))}</small>
        <p>${escapeHtml(compactText(candidate.caption || "无图注", 220))}</p>
      </div>`;
    }).join("")}</div>`;
  }
  output += renderAuditRows([
    { label: "生成图片描述", value: auditHasValue(timings.caption_seconds) ? `${timings.caption_seconds}s` : "", mono: true },
    { label: "Caption BM25", value: auditHasValue(timings.caption_bm25_seconds) ? `${timings.caption_bm25_seconds}s` : "", mono: true },
    { label: "Caption Dense", value: auditHasValue(timings.caption_dense_seconds) ? `${timings.caption_dense_seconds}s` : "", mono: true },
    { label: "DINOv2 图像向量", value: auditHasValue(timings.dinov2_seconds) ? `${timings.dinov2_seconds}s` : "", mono: true },
    { label: "Qwen-VL 原图复核", value: auditHasValue(timings.qwen_verify_seconds) ? `${timings.qwen_verify_seconds}s` : "", mono: true },
    { label: "三路检索总耗时", value: auditHasValue(timings.total_seconds) ? `${timings.total_seconds}s` : "", mono: true },
  ]);
  return output;
}

function auditEventGroup(stage, kind) {
  const value = String(stage || "").toLowerCase();
  if (["done", "complete"].includes(value)) return "done";
  if (value === "error") return "error";
  if (["model", "generate", "compose", "finalize"].includes(value)) return kind === "visual_manual" ? "result" : "model";
  if (["retrieve", "search", "tool", "knowledge", "rerank", "evidence"].includes(value)) return "retrieval";
  if (["vision", "images"].includes(value)) return "vision";
  if (["route", "scope", "classify", "classification"].includes(value)) return kind === "customer_service" ? "classify" : "route";
  return "intake";
}

function latestAuditEvent(rec, group, kind) {
  const events = Array.isArray(rec?.events) ? rec.events : [];
  return [...events].reverse().find((event) => auditEventGroup(event.stage, kind) === group) || null;
}

function auditTimingRows(trace, rec) {
  const timings = trace?.timings || {};
  const retrievalSeconds = auditHasValue(timings.retrieval_seconds)
    ? timings.retrieval_seconds
    : timings.retrieval_elapsed;
  const generationSeconds = auditHasValue(timings.generation_seconds)
    ? timings.generation_seconds
    : timings.generation_elapsed;
  return renderAuditRows([
    { label: "检索耗时", value: auditHasValue(retrievalSeconds) ? `${retrievalSeconds}s` : "", mono: true },
    { label: "首 Token", value: auditHasValue(timings.first_token_seconds) ? `${timings.first_token_seconds}s` : "", mono: true },
    { label: "原文核对", value: auditHasValue(timings.alignment_seconds) ? `${timings.alignment_seconds}s` : "", mono: true },
    { label: "生成耗时", value: auditHasValue(generationSeconds) ? `${generationSeconds}s` : "", mono: true },
    { label: "服务端总计", value: auditHasValue(timings.total_seconds) ? `${timings.total_seconds}s` : "", mono: true },
    { label: rec?.running ? "页面已耗时" : "页面总耗时", value: `${(Number(rec?.elapsed) || 0).toFixed(1)}s`, mono: true },
  ]);
}

function renderContextNarrative(historyAudit, context, entities, requested, applied, retrievalQuestion) {
  if (!requested) {
    return `<div class="audit-context-narrative">本轮以独立问答执行：没有发送历史 Context Packet，也不会继承前一轮产品、型号或部件。</div>`;
  }
  if (!applied) {
    return `<div class="audit-context-narrative">已请求多轮上下文，但当前会话没有可注入的历史实体；本轮仍按当前问题独立检索。</div>`;
  }
  const inherited = [
    entities.product && `产品“${entities.product}”`,
    entities.model && `型号“${entities.model}”`,
    (entities.component || context.component) && `部件“${entities.component || context.component}”`,
  ].filter(Boolean).join("、") || "已确认历史实体";
  const rewrite = context.applied && context.resolved_question
    ? `已将指代表达改写为“${context.resolved_question}”`
    : "当前问题无需指代改写";
  const query = retrievalQuestion ? `，并以“${compactText(retrievalQuestion, 180)}”进入检索` : "";
  return `<div class="audit-context-narrative">本轮继承${inherited}；${rewrite}${query}。</div>`;
}

function renderRetrievalChannel(trace, channel, scoreField, scoreLabel) {
  const candidates = Array.isArray(trace?.retrieval?.candidates) ? trace.retrieval.candidates : [];
  const rows = candidates
    .filter((item) => Number.isFinite(Number(item?.channel_ranks?.[channel])))
    .sort((left, right) => Number(left.channel_ranks[channel]) - Number(right.channel_ranks[channel]))
    .slice(0, 8);
  if (!rows.length) return `<div class="audit-inline-note">该通道没有返回独立排名。</div>`;
  return `<div class="audit-channel-list">${rows.map((item) => `<div class="audit-channel-row">
    <b>#${escapeHtml(String(item.channel_ranks[channel]))} · Chunk ${escapeHtml(String(item.chunk_id ?? "—"))}</b>
    <span>${escapeHtml(compactText(item.heading || "未命名片段", 150))}</span>
    <code>${escapeHtml(scoreLabel)}=${escapeHtml(auditNumber(item[scoreField], scoreField === "rrf_score" ? 5 : 4))}</code>
  </div>`).join("")}</div>`;
}

function renderHistoryContextTurns(turns) {
  const visible = Array.isArray(turns) ? turns.slice(-4) : [];
  if (!visible.length) return "";
  return `<div class="audit-context-turns"><b>注入的最近对话</b>${visible.map((turn, index) => {
    const role = String(turn?.role || "").toLowerCase() === "assistant" ? "助手" : "用户";
    return `<div><span>${String(index + 1).padStart(2, "0")} · ${role}</span><p>${escapeHtml(compactText(turn?.content || "", 220))}</p></div>`;
  }).join("")}</div>`;
}

function renderContextResolution(trace, rec) {
  const historyAudit = trace?.history_context || {};
  const packet = trace?.context_packet || {};
  const context = historyAudit.resolution || trace?.context_resolution || {};
  const entities = historyAudit.entities || packet.entities || {};
  const hasRequestedFlag = Object.prototype.hasOwnProperty.call(historyAudit, "requested");
  const requested = hasRequestedFlag
    ? Boolean(historyAudit.requested)
    : Boolean(Number(trace?.session_history_turns || rec?.contextTurns || 0) || auditHasValue(context.component));
  const applied = Object.prototype.hasOwnProperty.call(historyAudit, "applied")
    ? Boolean(historyAudit.applied)
    : Boolean(Number(trace?.session_history_turns || rec?.contextTurns || 0) || context.applied);
  const recentTurns = historyAudit.recent_turns || packet.recent_turns || [];
  const inheritedProduct = entities.product || trace?.route?.selected_manual || trace?.product_route?.products;
  const inheritedComponent = entities.component || context.component;
  const resolutionState = context.applied
    ? "已完成部件指代消解"
    : (requested ? "未发生指代改写（当前问题已明确或没有可继承部件）" : "未执行（多轮开关关闭）");
  const rows = renderAuditRows([
    { label: "执行节点", value: "context_packet.normalize → resolve_context_component_query", mono: true, wide: true },
    { label: "多轮开关请求", value: requested ? "开启" : "关闭" },
    { label: "本轮上下文注入", value: applied ? "已注入 Context Packet" : "未注入" },
    { label: "Packet 来源", value: historyAudit.source || (packet && Object.keys(packet).length ? "结构化 Context Packet" : "无历史 Packet") },
    { label: "Packet 版本", value: historyAudit.packet_version ?? trace?.context_packet_version ?? packet.version },
    { label: "服务端历史轮次", value: historyAudit.server_session_turns ?? trace?.session_history_turns ?? rec?.contextTurns ?? 0 },
    { label: "结构化用户轮次", value: historyAudit.structured_user_turns },
    { label: "历史摘要字符数", value: historyAudit.supplied_history_chars },
    { label: "继承产品", value: inheritedProduct },
    { label: "继承型号", value: entities.model },
    { label: "继承部件", value: inheritedComponent },
    { label: "用户原问", value: historyAudit.original_question || context.original_question || rec?.question, wide: true },
    { label: "消解状态", value: resolutionState, wide: true },
    { label: "消解规则", value: context.reason, mono: true, wide: true },
    { label: "指代改写", value: context.resolved_question, wide: true, mono: true },
    { label: "最终检索 Query", value: historyAudit.retrieval_question || trace?.query?.semantic, wide: true, mono: true },
  ]);
  return `${renderContextNarrative(historyAudit, context, entities, requested, applied, historyAudit.retrieval_question || trace?.query?.semantic)}${rows}${renderHistoryContextTurns(recentTurns)}`;
}

function renderSubjectScope(trace) {
  const focus = trace?.explicit_subject_focus || {};
  return renderAuditRows([
    { label: "代码节点", value: "focus_related_evidence_on_explicit_subject", mono: true, wide: true },
    { label: "范围门启用", value: auditHasValue(focus.applied) ? focus.applied : false },
    { label: "部件主体", value: focus.subject },
    { label: "判断原因", value: focus.reason },
    { label: "提升为部件核心证据", value: focus.promoted_core_chunk_ids },
    { label: "保留 Related", value: focus.selected_chunk_ids },
    { label: "按部件裁剪的证据", value: focus.projected_chunk_ids },
    { label: "拒绝 Related", value: focus.rejected_chunk_ids, wide: true },
    { label: "生成证据", value: trace?.generation_related_selected },
    { label: "生成前剔除", value: trace?.generation_related_rejected, wide: true },
  ]);
}

function buildManualAuditStages(trace, rec) {
  const query = trace.query || {};
  const route = trace.route || {};
  const productRoute = trace.product_route || {};
  const candidates = Array.isArray(trace.retrieval?.candidates) ? trace.retrieval.candidates : [];
  const selected = Array.isArray(trace.evidence?.selected) ? trace.evidence.selected : [];
  const binding = trace.structural_picture_binding || {};
  const alignment = trace.answer_evidence_alignment || {};
  const replay = isEvidenceReplay(trace);
  const routeProducts = route.candidates || productRoute.products || [];
  const detailedStages = [
    {
      key: "context",
      title: "多轮上下文与部件指代消解",
      detail: renderContextResolution(trace, rec),
    },
    {
      key: "sparse",
      title: "BM25 稀疏召回",
      detail: `${renderAuditRows([
        { label: "代码节点", value: "RetrievalEngine.search_manual / BM25Okapi", mono: true, wide: true },
        { label: "BM25 Query", value: query.sparse, wide: true, mono: true },
      ])}${renderRetrievalChannel(trace, "original_bm25", "bm25_raw", "BM25")}`,
    },
    {
      key: "dense",
      title: "Dense 向量召回",
      detail: `${renderAuditRows([
        { label: "代码节点", value: "embedding_client → FAISS cosine search", mono: true, wide: true },
        { label: "Dense Query", value: query.semantic || trace.visual_retrieval_query, wide: true, mono: true },
      ])}${renderRetrievalChannel(trace, "dense", "dense_cosine", "cosine")}`,
    },
    {
      key: "fusion",
      title: "RRF 融合与 Rerank 重排",
      detail: `${renderAuditRows([
        { label: "代码节点", value: "reciprocal_rank_fusion → rerank_client", mono: true, wide: true },
        { label: "候选数量", value: trace.retrieval?.candidate_count || candidates.length },
        { label: "过滤数量", value: trace.retrieval?.filtered_count || trace.retrieval_filtered_extremely_low || 0 },
      ])}${renderCandidateRanking(trace)}`,
    },
    {
      key: "scope",
      title: "部件范围门与 Related 证据裁剪",
      detail: renderSubjectScope(trace),
    },
  ];
  const stages = [
    {
      key: "intake",
      title: "问题解析",
      detail: renderAuditRows([
        { label: "原始问题", value: query.original || trace.original_query || rec.question, wide: true },
        { label: "BM25 Query", value: query.sparse, wide: true },
        { label: "Dense Query", value: query.semantic || trace.visual_retrieval_query, wide: true },
        { label: "规范化", value: trace.query_normalization?.normalized_question },
      ]),
    },
    {
      key: "route",
      title: "产品与手册锁定",
      detail: renderAuditRows([
        { label: "锁定手册", value: route.selected_manual || (routeProducts.length === 1 ? routeProducts[0] : "") },
        { label: "候选手册", value: routeProducts },
        { label: "锁定依据", value: auditRouteReason(route.reason || productRoute.reason) },
        { label: "置信度", value: route.confidence },
      ]),
    },
    {
      key: "retrieval",
      title: "BM25 + Dense + RRF 混合召回",
      detail: `<div class="audit-stage-summary">候选 ${Number(trace.retrieval?.candidate_count || candidates.length)} 个 · 过滤 ${Number(trace.retrieval?.filtered_count || trace.retrieval_filtered_extremely_low || 0)} 个 · 展示真实检索分数与重排结果</div>${renderCandidateRanking(trace)}`,
    },
    {
      key: "evidence",
      title: replay ? "检索证据取舍" : "证据取舍与模型输入",
      detail: `<div class="audit-stage-summary">最终检索证据 ${selected.length} 块；主证据优先，辅助片段受证据预算约束</div>${renderEvidenceSelection(trace)}`,
    },
    {
      key: "model",
      title: replay ? "审核答案输出" : "答案生成",
      detail: `${renderAuditRows([
        { label: "执行模式", value: trace.mode },
        { label: replay ? "生成模型" : "模型通道", value: replay ? "未调用" : trace.provider_route },
        { label: replay ? "输出方式" : "模型调用", value: replay ? "检索回放后直接输出已核对答案" : (trace.result?.turns ? `${trace.result.turns} 次生成` : "单次生成") },
      ])}${auditTimingRows(trace, rec)}`,
    },
  ];
  if (replay && alignment.applied) {
    stages.splice(4, 0, {
      key: "alignment",
      title: "答案与手册原文核对",
      detail: renderAnswerEvidenceAlignment(trace),
    });
  }
  const finalPics = binding.final_pics || trace.result?.pics || rec.images?.map((item) => item.name || item.file) || [];
  if (binding.anchor_count || finalPics.length) {
    stages.push({
      key: "images",
      title: "图文与来源整理",
      detail: renderAuditRows([
        { label: "正文图锚点", value: binding.anchor_count },
        { label: "最终图片", value: finalPics },
        { label: "绑定候选", value: binding.answer_bound_candidates || binding.candidates },
      ]),
    });
  }
  stages.push({
    key: "done",
    title: "处理完成",
    detail: renderAuditRows([
      { label: "执行链路", value: trace.execution_path || "lightweight_rag", mono: true },
      { label: "候选数量", value: candidates.length },
      { label: "证据数量", value: selected.length },
      { label: "输出图片", value: finalPics.length },
    ]),
  });
  const legacyRetrievalIndex = stages.findIndex((stage) => stage.key === "retrieval");
  if (legacyRetrievalIndex >= 0) stages.splice(legacyRetrievalIndex, 1);
  stages.splice(1, 0, detailedStages[0]);
  const routeStageIndex = stages.findIndex((stage) => stage.key === "route");
  stages.splice(routeStageIndex + 1, 0, ...detailedStages.slice(1));
  return stages;
}

function buildVisualAuditStages(trace, rec) {
  const media = trace.media_ingest || {};
  const visual = trace.visual_preroute || {};
  const manualInput = trace.manual_mode_input || {};
  const pageTitles = (media.page_contexts || []).map((item) => item.title).filter(Boolean);
  const mediaErrors = Array.isArray(media.errors) ? media.errors : [];
  const stages = [
    {
      key: "intake",
      title: "请求与媒体识别",
      detail: renderAuditRows([
        { label: "用户问题", value: trace.query?.original || trace.original_query || rec.question, wide: true },
        { label: "上传图片", value: auditHasValue(manualInput.has_image) ? manualInput.has_image : Number(trace.input_images_count || media.input_image_count || 0) > 0 },
        { label: "包含链接", value: auditHasValue(manualInput.has_link) ? manualInput.has_link : (media.discovered_urls || []).length > 0 },
      ]),
    },
    {
      key: "media",
      title: "图片 / 链接解析",
      detail: `${renderAuditRows([
        { label: "输入图片", value: media.input_image_count ?? trace.input_images_count ?? 0 },
        { label: "发现链接", value: (media.discovered_urls || []).length },
        { label: "链接取图", value: (media.fetched_images || []).length },
        { label: "成功解析", value: media.resolved_image_count ?? trace.resolved_images_count ?? 0 },
        { label: "页面标题", value: pageTitles, wide: true },
        { label: "解析错误", value: mediaErrors.length },
      ])}${mediaErrors.length ? `<details class="audit-flat-details"><summary>媒体解析错误详情</summary><div class="audit-dropped-list">${mediaErrors.map((item) => `<div><b>${escapeHtml(compactText(item.url || "媒体输入", 120))}</b><span>${escapeHtml(compactText(item.error || "读取失败", 220))}</span></div>`).join("")}</div></details>` : ""}`,
    },
    {
      key: "vision",
      title: "视觉识别",
      detail: `${renderAuditRows([
        { label: "识别策略", value: visual.strategy || "视觉预路由" },
        { label: "识别模型", value: visual.model || visual.provider },
        { label: "识别产品", value: visual.product },
        { label: "可见对象", value: visual.objects, wide: true },
        { label: "问题焦点", value: visual.focus, wide: true },
        { label: "检索意图", value: visual.intent, wide: true },
        { label: "置信度", value: visual.confidence },
        { label: "Terra 候选", value: visual.terra_candidate },
        { label: "耗时", value: auditHasValue(visual.elapsed_s) ? `${visual.elapsed_s}s` : "", mono: true },
      ])}${visual.error || visual.terra_error ? `<div class="audit-warning-line">视觉识别异常：${escapeHtml(compactText(visual.error || visual.terra_error, 260))}</div>` : ""}`,
    },
  ];
  if (visual.vector_trace || auditHasValue(visual.vector_score)) {
    stages.push({ key: "vector", title: "手册图片向量匹配", detail: renderVisualVector(trace) });
  }
  const hasInputImage = Boolean(manualInput.has_image || trace.input_images_count || media.input_image_count);
  const threeWayDetail = renderThreeWayImageRetrieval(trace, hasInputImage);
  if (threeWayDetail) {
    stages.push({ key: "image-rrf", title: "图片三路召回与原图复核", detail: threeWayDetail });
  }
  stages.push({
    key: "route",
    title: "产品与手册判定",
    detail: renderAuditRows([
      { label: "最终产品", value: visual.product || trace.route?.selected_manual || trace.product_route?.products },
      { label: "判定方式", value: visual.decision || visual.manual_grounding || trace.route?.reason },
      { label: "图片命中", value: visual.manual_image_match?.image_id },
      { label: "命中章节", value: visual.manual_image_match?.heading },
    ]),
  });
  if (Array.isArray(trace.retrieval?.candidates) && trace.retrieval.candidates.length) {
    stages.push({
      key: "retrieval",
      title: "手册混合召回",
      detail: renderCandidateRanking(trace),
    });
    stages.push({
      key: "evidence",
      title: "证据与图文绑定",
      detail: renderEvidenceSelection(trace),
    });
  }
  const productOnly = String(trace.execution_path || "").includes("image_product_dual");
  const linkClarification = String(trace.execution_path || "").includes("link_only_clarification");
  stages.push({
    key: "result",
    title: productOnly ? "识别结果整理" : (linkClarification ? "链接处理结果" : "答案生成与配图"),
    detail: `${productOnly ? `<div class="audit-inline-note">本次链路只完成产品 / 手册识别；未执行 BM25、Dense、RRF 或手册答案生成。</div>` : ""}${linkClarification ? `<div class="audit-inline-note">链接未解析成可用图片，系统返回澄清问题；未执行手册检索。</div>` : ""}${renderAuditRows([
      { label: "执行链路", value: trace.execution_path, mono: true },
      { label: "输出图片", value: trace.structural_picture_binding?.final_pics || trace.result?.pics },
      { label: "模型调用", value: trace.result?.turns },
    ])}${auditTimingRows(trace, rec)}`,
  });
  stages.push({
    key: "done",
    title: "处理完成",
    detail: renderAuditRows([
      { label: "识别产品", value: visual.product || "未唯一确定" },
      { label: "解析图片", value: media.resolved_image_count ?? trace.resolved_images_count ?? 0 },
      { label: "向量候选", value: visual.vector_trace?.hits?.length ?? 0 },
    ]),
  });
  return stages;
}

function buildServiceAuditStages(trace, rec) {
  const classifier = trace.classifier || {};
  const turns = trace.session_history_turns ?? rec.contextTurns ?? 0;
  return [
    {
      key: "intake",
      title: "问题解析",
      detail: renderAuditRows([
        { label: "用户问题", value: trace.query?.original || trace.original_query || rec.question, wide: true },
        { label: "输入规范化", value: trace.query_normalization?.normalized_question },
      ]),
    },
    {
      key: "classify",
      title: "客服意图识别",
      detail: `${renderAuditRows([
        { label: "分类结果", value: classifier.route || "service" },
        { label: "判断方式", value: classifier.strategy || classifier.kind || "本地规则" },
        { label: "判断耗时", value: auditHasValue(classifier.elapsed) ? `${classifier.elapsed}s` : "", mono: true },
        { label: "手册检索", value: "未执行（客服题不需要 BM25 / Dense / RRF）", wide: true },
      ])}`,
    },
    {
      key: "context",
      title: "历史上下文",
      detail: renderAuditRows([
        { label: "历史轮数", value: Number(turns) },
        { label: "上下文版本", value: trace.context_packet_version },
        { label: "使用状态", value: Number(turns) > 0 ? "已使用当前客服事项的历史上下文" : "未使用历史上下文" },
      ]),
    },
    {
      key: "model",
      title: "客服回答生成",
      detail: `${renderAuditRows([
        { label: "执行模式", value: trace.mode || "history_aware_single_generation" },
        { label: "模型通道", value: trace.provider_route },
        { label: "工具调用", value: trace.result?.tool_calls ?? 0 },
        { label: "生成轮次", value: trace.result?.turns ?? 1 },
      ])}${auditTimingRows(trace, rec)}`,
    },
    {
      key: "done",
      title: "处理完成",
      detail: renderAuditRows([
        { label: "执行链路", value: trace.execution_path || "lightweight_service", mono: true },
        { label: "回答类型", value: "纯客服回答" },
        { label: "引用手册", value: "0 本" },
      ]),
    },
  ];
}

function auditCurrentStageIndex(stages, trace, rec, kind) {
  if (!rec?.running) return stages.length - 1;
  const events = Array.isArray(rec?.events) ? rec.events : [];
  const latest = events.length ? events[events.length - 1] : null;
  let currentKey = auditEventGroup(latest?.stage, kind);
  if (kind === "manual_rag" && Array.isArray(trace?.evidence?.selected)) currentKey = "evidence";
  if (kind === "visual_manual") {
    const media = trace?.media_ingest || {};
    const hasMediaTrace = auditHasValue(media.input_image_count)
      || auditHasValue(media.resolved_image_count)
      || auditHasValue(media.discovered_urls)
      || auditHasValue(media.errors);
    const visual = trace?.visual_preroute || {};
    if (hasMediaTrace) currentKey = "media";
    if (auditHasValue(visual.product) || auditHasValue(visual.strategy) || visual.used === true) currentKey = "vision";
    if (visual.vector_trace) currentKey = "vector";
  }
  if (kind === "customer_service" && trace?.classifier) currentKey = "classify";
  const eventKey = auditEventGroup(latest?.stage, kind);
  const currentIndex = stages.findIndex((stage) => stage.key === currentKey);
  const eventIndex = stages.findIndex((stage) => stage.key === eventKey);
  return Math.max(0, currentIndex, eventIndex);
}

function renderProtectedAuditTimeline(rec) {
  if (!rec || rec.id === "__idle__") {
    return `<div class="log-empty">提交问题后，这里会实时显示可审计的路由、检索、证据、视觉处理与生成过程。</div>`;
  }
  const trace = rec?.audit && typeof rec.audit === "object" ? rec.audit : {};
  const kind = auditViewKind(trace, rec);
  const meta = AUDIT_VIEW_META[kind];
  const stages = kind === "customer_service"
    ? buildServiceAuditStages(trace, rec)
    : (kind === "visual_manual" ? buildVisualAuditStages(trace, rec) : buildManualAuditStages(trace, rec));
  const currentIndex = auditCurrentStageIndex(stages, trace, rec, kind);

  const timeline = stages.map((stage, index) => {
    const stateClass = stage.key === "error"
      ? "err"
      : (index < currentIndex || !rec.running ? "done" : (index === currentIndex ? "active" : "pending"));
    const stateLabel = stateClass === "done" ? "已完成" : (stateClass === "active" ? "进行中" : "等待");
    const event = latestAuditEvent(rec, stage.key, kind);
    const message = event ? humanizeProgress(event.message, event.stage) : "";
    const elapsed = Number.isFinite(Number(event?.elapsed)) ? `${Number(event.elapsed).toFixed(1)}s` : "";
    return `<div class="tl-item tl-${stateClass} stage-${escapeHtml(stage.key)}">
      <span class="tl-dot" aria-hidden="true"></span>
      <div class="tl-body">
        <div class="tl-heading"><span class="tl-seq">${String(index + 1).padStart(2, "0")}</span><b>${escapeHtml(stage.title)}</b><span class="audit-stage-state">${stateLabel}</span>${elapsed ? `<time>${elapsed}</time>` : ""}</div>
        ${message && stateClass !== "pending" ? `<div class="tl-msg">${escapeHtml(message)}</div>` : ""}
        ${stateClass !== "pending" || !rec.running ? stage.detail : ""}
      </div>
    </div>`;
  }).join("");

  return `<section class="audit-flow audit-flow-${escapeHtml(kind)}" data-audit-kind="${escapeHtml(kind)}">
    <header class="audit-flow-head"><div><b>${escapeHtml(meta.label)}</b><span>${escapeHtml(meta.description)}</span></div><code>${escapeHtml(trace.execution_path || (rec.running ? "live" : "completed"))}</code></header>
    <div class="audit-timeline">${timeline}</div>
  </section>`;
}

function renderProcess(rec) {
  // Paint a record into the fixed sidebar DOM nodes.
  const r = rec || IDLE_PROCESS;
  if (els.progressStage) {
    els.progressStage.dataset.state = r.kind === "error"
      ? "error"
      : (r.running ? "running" : (r.kind === "api" ? "done" : "idle"));
  }
  if (els.progressPanel) els.progressPanel.classList.toggle("active", !!r.running);
  if (els.progressStatusCard) els.progressStatusCard.classList.toggle("active", !!r.running);
  const viewKind = auditViewKind(r.audit, r);
  if (els.logbar) els.logbar.dataset.auditKind = viewKind;
  if (els.progressStage) els.progressStage.textContent = r.stage || "待命";
  if (els.progressStatus) {
    els.progressStatus.textContent = humanizeProgress(r.status) || r.status || "";
    els.progressStatus.classList.toggle("api-success", r.kind === "api");
    els.progressStatus.classList.toggle("api-error", r.kind === "error");
  }
  if (els.progressElapsed) els.progressElapsed.textContent = `${(Number(r.elapsed) || 0).toFixed(1)}s`;
  if (els.progressPercent) els.progressPercent.textContent = `${Math.round(Math.max(0, Math.min(100, Number(r.percent) || 0)))}%`;
  if (els.progressBar) {
    els.progressBar.style.width = `${Math.max(0, Math.min(100, Number(r.percent) || 0))}%`;
    els.progressBar.classList.toggle("running", !!r.running);
  }

  // Result summary chips (mode / manuals / image count / source)
  if (els.progressSummary) {
    const s = r.summary;
    const hasProcess = r.id && r.id !== "__idle__";
    if (s || hasProcess) {
      const chips = [];
      const meta = AUDIT_VIEW_META[viewKind] || AUDIT_VIEW_META.manual_rag;
      chips.push(`<span class="sum-chip sum-${viewKind}">${escapeHtml(meta.label)}</span>`);
      if (Number(s?.imageCount) > 0) chips.push(`<span class="sum-chip">图 ${s.imageCount}</span>`);
      if (Number(s?.manualCount) > 0) chips.push(`<span class="sum-chip">手册 ${s.manualCount} 本</span>`);
      if (s?.elapsed) chips.push(`<span class="sum-chip sum-time">用时 ${s.elapsed}</span>`);
      els.progressSummary.innerHTML = chips.join("");
      els.progressSummary.hidden = chips.length === 0;
    } else {
      els.progressSummary.hidden = true;
      els.progressSummary.innerHTML = "";
    }
  }

  // Image thumbnails (like GPT showing the figures it used)
  if (els.progressImages && els.progressThumbs) {
    const imgs = r.images || [];
    const base = state.data && state.data.imageBase;
    if (imgs.length && base) {
      els.progressThumbs.innerHTML = imgs
        .map((img, i) => {
          const file = encodeURIComponent(img.file || img.name || "");
          const name = escapeHtml(img.name || img.file || "");
          return `<figure class="thumb"><span class="thumb-idx">${i + 1}</span><img src="${base}${file}" alt="${name}" loading="lazy"><figcaption>${name}</figcaption></figure>`;
        })
        .join("");
      els.progressImages.hidden = false;
    } else {
      els.progressImages.hidden = true;
      els.progressThumbs.innerHTML = "";
    }
  }

  // RAG thinking timeline: humanize + collapse consecutive duplicates, then
  // render as a vertical timeline (dots + connector). The last step is "active"
  // with a spinner while the request runs; earlier steps show as done.
  if (els.progressLog) {
    const timeline = renderProtectedAuditTimeline(r);
    els.progressLog.innerHTML = timeline || `<div class="log-empty">提问后，这里会实时展示问题解析、手册锁定、混合召回、证据选择和答案生成。</div>`;
    if (r.running) els.progressLog.scrollTop = els.progressLog.scrollHeight;
    else els.progressLog.scrollTop = 0;
  }
}

function liveTouch() {
  // Repaint only if the in-flight record is the one being viewed.
  if (state.live && state.activeProcessId === state.live.id) renderProcess(state.live);
}

function setActiveProcess(id) {
  // Switch the sidebar to show a specific question's stored RAG flow.
  const rec = state.processes.get(id);
  if (!rec) return;
  state.activeProcessId = id;
  renderProcess(rec);
  for (const m of document.querySelectorAll(".message.process-active")) m.classList.remove("process-active");
  const el = document.querySelector(`.message[data-proc-id="${id}"]`);
  if (el) el.classList.add("process-active");
  if (window.matchMedia("(max-width: 820px)").matches && els.appShell) {
    openMobileLogbar();
  }
}

function renderIdleProcess() {
  state.activeProcessId = IDLE_PROCESS.id;
  renderProcess(IDLE_PROCESS);
}

function bindProcessToMessage(wrap, id) {
  // Make a chat turn clickable: selecting it shows that question's RAG flow in
  // the right sidebar (ChatGPT-style — one question's process at a time).
  if (!wrap || !id) return;
  wrap.dataset.procId = id;
  wrap.classList.add("has-process");
  wrap.title = "点击查看该问答的 RAG 思考过程";
  wrap.addEventListener("click", () => setActiveProcess(id));
}

function resetProgressTerminal() {
  if (state.live) state.live.events = [];
  liveTouch();
}

function appendProgressTerminal(stage, message, elapsed) {
  if (!state.live) return;
  state.live.events.push({
    stage: stage || "info",
    message: String(message || "").trim(),
    elapsed: Number.isFinite(Number(elapsed)) ? Number(elapsed) : null,
  });
  liveTouch();
}

function setProgressStage(text) {
  if (state.live) state.live.stage = text || "处理中";
  liveTouch();
}

function setProgressElapsed(seconds) {
  if (state.live) state.live.elapsed = Number.isFinite(Number(seconds)) ? Number(seconds) : 0;
  liveTouch();
}

function setProgressPercent(value) {
  if (state.live) state.live.percent = Math.max(0, Math.min(100, Number(value) || 0));
  liveTouch();
}

function progressStageName(text) {
  const value = String(text || "");
  if (value.includes("提交")) return "提交中";
  if (value.includes("检索") || value.includes("證") || value.includes("璇")) return "检索中";
  if (value.includes("生成") || value.includes("model") || value.includes("智能")) return "模型生成";
  if (value.includes("整理") || value.includes("渲染")) return "后处理";
  return "处理中";
}

function progressStageLabel(stage) {
  const key = String(stage || "").toLowerCase();
  const labels = {
    start: "开始",
    accepted: "接收",
    input: "输入",
    analyze: "问题分析",
    policy: "策略",
    scope: "范围",
    connected: "连接",
    validated: "校验",
    route: "路由",
    retrieve: "检索",
    knowledge: "检索",
    evidence: "证据",
    rerank: "重排",
    grounding: "约束",
    model: "模型",
    compose: "答案生成",
    postprocess: "后处理",
    images: "图片",
    customer: "客服",
    vision: "视觉",
    error: "错误",
    done: "完成",
    complete: "完成",
    system: "系统",
    info: "信息",
  };
  return labels[key] || stage || "信息";
}


function compactText(value, limit) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function knownProductNames() {
  return (state.data?.products || [])
    .map((product) => String(product.name || "").trim())
    .filter(Boolean)
    .sort((left, right) => right.length - left.length);
}

// Product aliases are catalog identity, not question-specific answer rules.
// They let an English question override a stale Chinese quick-product
// selection. An empty `product` means the corpus has the manual but the UI has
// no dedicated category; it is still explicit so the previous scope is cleared.
const QUESTION_PRODUCT_ALIASES = [
  { product: "空气炸锅", aliases: ["air fryer", "airfryer"] },
  { product: "摩托艇", aliases: ["boat", "personal watercraft"] },
  { product: "相机", aliases: ["camera"] },
  { product: "耳机", aliases: ["earphones", "earbuds"] },
  { product: "咖啡机", aliases: ["espresso machine", "coffee machine", "coffee maker"] },
  { product: "烤架", aliases: ["gas grill", "grill"] },
  { product: "割草机", aliases: ["lawn mower"] },
  { product: "电子阅读器", aliases: ["media player", "e-reader", "ereader", "ebook reader"] },
  { product: "微波炉", aliases: ["microwave"] },
  { product: "主板", aliases: ["motherboard"] },
  { product: "固定电话", aliases: ["landline", "cordless phone", "base station"] },
  { product: "传真机", aliases: ["printer", "fax", "fax machine"] },
  { product: "雪地摩托", aliases: ["snowmobile"] },
  { product: "电视/天线", aliases: ["tv", "television"] },
  { product: "电动牙刷", aliases: ["toothbrush", "electric toothbrush", "sonicare"] },
  { product: "VR头显", aliases: ["vr headset"] },
  { product: "扫地机器人", aliases: ["vacuum", "vacuum cleaner", "robot vacuum", "robotic vacuum", "roomba"] },
  { product: "水上摩托", aliases: ["waverunner", "jet ski", "jetski", "pwc"] },
  { product: "人体工学椅", aliases: ["ergonomic chair"] },
  { product: "健身单车", aliases: ["exercise bike", "exercise bicycle"] },
  { product: "健身追踪器", aliases: ["fitness tracker"] },
  { product: "儿童电动摩托车", aliases: ["kids electric scooter", "electric scooter"] },
  { product: "冰箱", aliases: ["refrigerator", "fridge"] },
  { product: "功能键盘", aliases: ["keyboard"] },
  { product: "发电机", aliases: ["generator"] },
  { product: "可编程温控器", aliases: ["programmable temperature controller", "thermostat"] },
  { product: "吹风机", aliases: ["leaf blower", "blower"] },
  { product: "水泵", aliases: ["water pump"] },
  { product: "洗碗机", aliases: ["dishwasher"] },
  { product: "烤箱", aliases: ["oven"] },
  { product: "电钻", aliases: ["power drill", "drill"] },
  { product: "混合即时相机", aliases: ["camera manual", "hybrid instant camera", "instant camera"] },
  { product: "空气净化器", aliases: ["air purifier"] },
  { product: "空调", aliases: ["air conditioner", "air conditioning"] },
  { product: "蒸汽清洁机", aliases: ["steam cleaner", "steam mop"] },
  { product: "蓝牙鼠标", aliases: ["bluetooth laser mouse", "laser mouse"] },
  // These English manuals currently have no dedicated UI category. They must
  // still block stale product inheritance and let the backend canonical router
  // choose the retrieval scope.
  { product: "", aliases: ["pressure cooker", "washing machine", "washer", "phone"] },
];

const QUESTION_PRODUCT_ALIAS_MATCH = QUESTION_PRODUCT_ALIASES.flatMap((entry) => (
  entry.aliases.map((alias) => ({ ...entry, alias }))
));

function resolveQuestionProduct(question) {
  const text = String(question || "").toLowerCase();
  const matches = knownProductNames().filter((name) => text.includes(name.toLowerCase()));
  const knownNames = new Set(knownProductNames());
  const aliasMatches = QUESTION_PRODUCT_ALIAS_MATCH
    .filter((entry) => entry.product && knownNames.has(entry.product))
    .filter((entry) => {
      const escaped = entry.alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`(?:^|[^a-z0-9])${escaped}(?:$|[^a-z0-9])`, "i").test(text);
    })
    .sort((left, right) => right.alias.length - left.alias.length);
  // Prefer a specific phrase such as "hybrid instant camera" over its shorter
  // generic "camera" match, otherwise one question becomes falsely ambiguous.
  const specificAliasMatches = aliasMatches.filter((entry) => !aliasMatches.some((other) => (
    other !== entry
    && other.alias.length > entry.alias.length
    && other.alias.includes(entry.alias)
  )));
  const allMatches = [...new Set([
    ...matches,
    ...specificAliasMatches.map((entry) => entry.product),
  ])];
  const hasUnmappedEnglishManual = QUESTION_PRODUCT_ALIAS_MATCH.some((entry) => {
    if (entry.product) return false;
    const escaped = entry.alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`(?:^|[^a-z0-9])${escaped}(?:$|[^a-z0-9])`, "i").test(text);
  });
  if (!allMatches.length) {
    return hasUnmappedEnglishManual
      ? { product: "", ambiguous: false, explicit: true }
      // The highlighted product is only a UI navigation state.  It is not
      // evidence about the user's current question and must never narrow a
      // retrieval request.  Otherwise a previous Generator question can make
      // an unrelated Boat question search only the Generator manual.
      : { product: "", ambiguous: false, explicit: false };
  }
  const distinct = [...new Set(allMatches.filter((name) => !allMatches.some((other) => (
    other !== name && other.includes(name)
  ))))];
  if (distinct.length !== 1) return { product: "", ambiguous: true, explicit: true };
  return { product: distinct[0], ambiguous: false, explicit: true };
}

function memoryTurnsFor(product) {
  return product ? (state.productMemories.get(product) || []) : [];
}

function buildHistoryContext(product) {
  // The UI keeps a short rolling summary instead of sending the full transcript.
  // It is meant to resolve follow-ups like "刚才那个步骤再说细一点", while the
  // current input remains the user's visible question.
  const turns = memoryTurnsFor(product).slice(-MAX_HISTORY_TURNS);
  if (!turns.length) return "";
  const lines = turns.map((turn, index) => {
    const product = turn.product ? `产品：${turn.product}；` : "";
    const mode = turn.modeLabel ? `模式：${turn.modeLabel}；` : "";
    const question = compactText(turn.question, 160);
    const answer = compactText(turn.answer, 260);
    return `${index + 1}. ${product}${mode}用户：${question}\n   助手：${answer}`;
  });
  return compactText(lines.join("\n"), MAX_HISTORY_CONTEXT_CHARS);
}

function extractHistoryConstraints(question) {
  return String(question || "")
    .split(/[。！？!?；;\n]+/)
    .map((part) => compactText(part, 220))
    .filter((part) => part && /(?:只|仅|不要|不得|必须|无需|不用|优先|禁止)/.test(part))
    .slice(0, 8);
}

function buildStructuredContext(product, currentQuestion) {
  const turns = memoryTurnsFor(product).slice(-4);
  if (!turns.length) return {};
  const recentTurns = [];
  const mediaFacts = [];
  const constraints = [];
  turns.forEach((turn, index) => {
    if (turn.question) recentTurns.push({ role: "user", content: compactText(turn.question, 420) });
    if (turn.answer) recentTurns.push({ role: "assistant", content: compactText(turn.answer, 420) });
    constraints.push(...extractHistoryConstraints(turn.question));
    const imageDescriptions = turn.imageDescriptions || [];
    const confidenceMatch = imageDescriptions.join(" ").match(/视觉识别置信度[：:]\s*(high|medium|low)/i);
    const visualConfidence = confidenceMatch ? confidenceMatch[1].toLowerCase() : "medium";
    imageDescriptions.forEach((fact) => {
      const text = compactText(fact, 360);
      if (text) {
        mediaFacts.push({
          fact: text,
          source: `turn_${index + 1}.image`,
          confidence: visualConfidence,
        });
      }
    });
  });
  const modelMatch = turns
    .map((turn) => turn.question || "")
    .join(" ")
    .match(/\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b/i);
  const entities = { product };
  if (modelMatch) entities.model = modelMatch[0].toUpperCase();
  const uniqueConstraints = [...new Set(constraints)].slice(0, 8);
  return {
    version: CONTEXT_PACKET_VERSION,
    summary: compactText(
      `产品：${product}；最近问题：${turns.at(-1)?.question || ""}`,
      700,
    ),
    entities,
    media_facts: mediaFacts.slice(-12),
    user_constraints: uniqueConstraints,
    recent_turns: recentTurns.slice(-8),
    retrieval_hint: HISTORY_ONLY_RE.test(String(currentQuestion || "")) ? "history_only" : "auto",
  };
}

function updateHistoryContextIndicator(product = state.activeProduct) {
  const enabled = Boolean(els.historyContextToggle?.checked);
  const count = memoryTurnsFor(product).length;
  if (els.contextModeBadge) {
    els.contextModeBadge.textContent = enabled ? "多轮上下文" : "独立问答";
    els.contextModeBadge.title = enabled
      ? "已开启：同一会话内继承当前产品与部件主体"
      : "已关闭：每个问题独立检索，不引用上一轮内容";
  }
  if (els.historyContextCount) els.historyContextCount.textContent = `${Math.min(count, MAX_HISTORY_TURNS)}轮`;
  if (els.historyContextScope) els.historyContextScope.textContent = `当前产品：${product || "未选择"}`;
  if (els.clearHistoryContext) els.clearHistoryContext.disabled = !product || !count || state.busy;
}

function rememberConversationTurn(question, item) {
  const product = item.contextProduct || item.product || state.activeProduct || "";
  if (!product || !knownProductNames().includes(product)) return;
  const turns = memoryTurnsFor(product);
  turns.push({
    question,
    answer: item.answer || "",
    product,
    modeLabel: item.modeLabel || (item.answerMode === "customer" ? "客服模式" : "手册模式"),
    imageDescriptions: item.imageDescriptions || [],
  });
  state.productMemories.set(product, turns.slice(-MAX_HISTORY_TURNS));
  persistProductMemories();
  updateHistoryContextIndicator(product);
}

async function readSseResponse(res, onDelta, onStatus, onAudit) {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("当前浏览器无法读取流式响应");
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalData = null;

  function consumeFrame(frame) {
    const lines = frame.split(/\r?\n/);
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const data = JSON.parse(dataLines.join("\n"));
    if (event === "delta") onDelta?.(String(data.text || ""));
    if (event === "status") onStatus?.(data);
    if (event === "audit") onAudit?.(data.retrieval_trace || data.trace || data);
    if (event === "done") finalData = data;
    if (event === "error") throw new Error(data.message || "流式生成失败");
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || "";
    for (const frame of frames) consumeFrame(frame);
    if (done) break;
  }
  if (buffer.trim()) consumeFrame(buffer);
  if (!finalData) throw new Error("流式连接结束，但没有收到最终答案");
  return { code: 0, msg: "success", data: finalData };
}

function sleepForDisplay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, Math.max(0, ms)));
}

async function simulateCustomerAnswerStream(loadingBubble, answer, startedAt) {
  const text = String(answer || "");
  const elapsed = performance.now() - startedAt;
  if (!text || elapsed >= CUSTOMER_STREAM_MIN_MS) return;

  const characters = Array.from(text);
  const targetMs = Math.min(
    CUSTOMER_STREAM_MAX_MS,
    Math.max(CUSTOMER_STREAM_MIN_MS, CUSTOMER_STREAM_TARGET_MS + Math.min(1_200, characters.length * 2)),
  );
  const remainingMs = Math.max(0, targetMs - elapsed);
  if (!remainingMs) return;

  loadingBubble.innerHTML = "";
  const liveAnswer = document.createElement("div");
  liveAnswer.className = "answer-text streaming-answer customer-streaming-answer";
  loadingBubble.appendChild(liveAnswer);
  setProgressStage("生成客服答复");
  setProgressStatus("客服正在整理并逐步呈现答复…");
  appendProgressTerminal("compose", "客服答复正在流式呈现", elapsed / 1000);

  const chunkCount = Math.min(120, Math.max(18, Math.ceil(characters.length / 12)));
  const chunkSize = Math.max(1, Math.ceil(characters.length / chunkCount));
  const intervalMs = remainingMs / Math.ceil(characters.length / chunkSize);
  for (let offset = 0; offset < characters.length; offset += chunkSize) {
    liveAnswer.textContent += characters.slice(offset, offset + chunkSize).join("");
    els.messages.scrollTop = els.messages.scrollHeight;
    if (offset + chunkSize < characters.length) await sleepForDisplay(intervalMs);
  }
  const displayedElapsed = performance.now() - startedAt;
  if (displayedElapsed < CUSTOMER_STREAM_MIN_MS) {
    await sleepForDisplay(CUSTOMER_STREAM_MIN_MS - displayedElapsed);
  }
}

async function simulateManualAnswerStream(loadingBubble, answer, startedAt) {
  // Exact/recommended answers do not receive upstream delta events because the
  // gateway replaces the model draft with the reviewed table entry at `done`.
  // Keep those manual answers visibly streamed just like live RAG answers.
  const text = String(answer || "");
  const elapsed = performance.now() - startedAt;
  if (!text || elapsed >= MANUAL_STREAM_MAX_MS) return;

  const characters = Array.from(text);
  const targetMs = Math.min(
    MANUAL_STREAM_MAX_MS,
    Math.max(MANUAL_STREAM_MIN_MS, MANUAL_STREAM_TARGET_MS + Math.min(1_600, characters.length * 2)),
  );
  const remainingMs = Math.max(0, targetMs - elapsed);
  if (!remainingMs) return;

  loadingBubble.innerHTML = "";
  const liveAnswer = document.createElement("div");
  liveAnswer.className = "answer-text streaming-answer manual-streaming-answer";
  loadingBubble.appendChild(liveAnswer);
  setProgressStage("生成手册答复");
  setProgressStatus("正在依据手册证据逐段呈现答复…");
  appendProgressTerminal("compose", "手册答复正在流式呈现", elapsed / 1000);

  const chunkCount = Math.min(110, Math.max(16, Math.ceil(characters.length / 18)));
  const chunkSize = Math.max(1, Math.ceil(characters.length / chunkCount));
  const intervalMs = remainingMs / Math.ceil(characters.length / chunkSize);
  for (let offset = 0; offset < characters.length; offset += chunkSize) {
    liveAnswer.textContent += characters.slice(offset, offset + chunkSize).join("");
    els.messages.scrollTop = els.messages.scrollHeight;
    if (offset + chunkSize < characters.length) await sleepForDisplay(intervalMs);
  }
}

async function callRealApi(question, requestId, onDelta, onStatus, onAudit, submittedAttachment = null) {
  // Build the official `/chat` request. The browser never calls the external LLM
  // provider directly and never sees a provider API key; it only talks to the
  // local `server.py` proxy with the demo Bearer Token.
  const remoteMedia = hasRemoteMediaUrl(question);
  const requestAttachment = submittedAttachment || state.attachment;
  if (requestAttachment) {
    setProgressStatus("正在编码上传图片，准备提交多模态请求...");
  } else if (remoteMedia) {
    setProgressStatus("检测到图片链接，正在安全下载并准备多模态解析...");
  }
  const images = requestAttachment ? [await readFileAsDataUrl(requestAttachment.file)] : [];
  const sessionId = state.sessionId || createId("raysource_memory");
  state.sessionId = sessionId;
  window.localStorage.setItem("ragv6_session_id", sessionId);
  requestId = requestId || createId("kf_req");
  const resolution = resolveQuestionProduct(question);
  const explicitProduct = resolution.product;
  // An implicit product from the previous turn may be used only after the
  // user turns history on. Without that opt-in, UI selection/history remains
  // non-evidence and this request is routed independently.
  const rememberedProduct = els.historyContextToggle?.checked && !resolution.ambiguous && !explicitProduct
    ? state.activeProduct
    : "";
  const contextProduct = explicitProduct || rememberedProduct || "";
  const customerServiceQuestion = isCustomerServiceQuestion(question);
  // Keep all website requests on the streaming gateway. This preserves the
  // audit contract for customer-service recommendations as well as manuals.
  const chatEndpoint = CHAT_ENDPOINT;
  // The legacy gateway may keep a last-product value for a session.  When
  // this question has no product fact of its own, use a request-scoped ID so
  // that a prior product cannot silently become its retrieval constraint.
  const upstreamSessionId = contextProduct
    ? sessionId
    : createId("raysource_unscoped");
  const useHistoryContext = Boolean(els.historyContextToggle?.checked && contextProduct && !resolution.ambiguous);
  const historyContext = useHistoryContext ? buildHistoryContext(contextProduct) : "";
  const contextPacket = useHistoryContext ? buildStructuredContext(contextProduct, question) : {};
  const timeoutMs = images.length || remoteMedia ? MULTIMODAL_TIMEOUT_MS : TEXT_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let payload;
  let res;
  try {
    // AbortController implements the frontend timeout. The backend also has its
    // own timeout, but having a browser-side guard keeps the UI responsive even
    // if the local server or network layer stalls.
    res = await fetch(chatEndpoint, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${CHAT_API_TOKEN}`,
        "X-Request-Id": requestId,
        "X-Client-Type": "web",
        "X-RAG-Channel": "web",
        "X-RAG-Session-Id": upstreamSessionId,
      },
      body: JSON.stringify({
        question,
        forced_product: (explicitProduct || (useHistoryContext ? contextProduct : "")) || undefined,
        model: state.activeModelProfile?.model || "gpt-5.6-terra",
        reasoning_effort: state.reasoningEffort || undefined,
        images,
        session_id: upstreamSessionId,
        request_id: requestId,
        stream: true,
        use_history_context: useHistoryContext,
        history_context: historyContext,
        context_packet: contextPacket,
        history_product: contextProduct,
        memory_epoch: state.productMemoryEpochs.get(contextProduct) || "base",
        // Deliberately do not send UI selection as routing evidence.
      }),
    });
    const contentType = String(res.headers.get("content-type") || "").toLowerCase();
    if (contentType.includes("text/event-stream")) {
      payload = await readSseResponse(res, onDelta, onStatus, onAudit);
    } else {
      const responseText = await res.text();
      try {
        payload = responseText ? JSON.parse(responseText) : {};
      } catch (_jsonError) {
        const snippet = responseText.replace(/\s+/g, " ").slice(0, 120);
        throw new Error(`后端/网关返回了非 JSON 内容：${snippet || "empty response"}`);
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`接口超时：${Math.round(timeoutMs / 1000)} 秒内未返回`);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
  if (!res.ok || payload.code !== 0) {
    throw new Error(payload.msg || payload.error || "API request failed");
  }

  const data = payload.data || {};

  // Deliberately do not persist the returned session identifier. Account
  // history is a display/archive feature; every RAG request remains independent.

  return {
    requestId: data.request_id || res.headers.get("x-request-id") || requestId,
    source: data.source || "api",
    answerMode: data.answer_mode || "manual",
    modeLabel: data.mode_label || (data.answer_mode === "customer" ? "客服模式" : "手册模式"),
    routing: data.routing || {},
    answerConfidence: data.answer_confidence || data.retrieval_trace?.answer_confidence || null,
    product: data.product || state.activeProduct || "产品",
    question,
    answer: data.answer || "",
    images: data.images || [],
    manuals: data.manuals || [],
    sources: data.sources || [],
    // Always derive the mode label locally. The backend's `mode_label` has been
    // observed to arrive mojibake-encoded (e.g. "手册模式" → "鎵嬪唽妯″紡"); the
    // UI keeps a clean Chinese label keyed off the reliable `answer_mode` field.
    modeLabel: data.answer_mode === "customer" ? "客服模式" : "手册模式",
    imageDescriptions: data.image_descriptions || [],
    historyContextUsed: Boolean(data.history_context_used),
    contextProduct: data.context_product || contextProduct || "",
    contextTurns: Number(data.context_turns || 0),
    contextSwitched: Boolean(data.context_switched),
    contextPacketVersion: Number(data.context_packet_version || 0),
    contextRetrievalHint: data.context_retrieval_hint || "auto",
    retrievalTrace: data.retrieval_trace
      || (Array.isArray(data.sources) ? data.sources.find((source) => source?.retrieval_trace)?.retrieval_trace : null)
      || null,
  };
}

async function handleSubmit(event) {
  // Main submit coordinator. It disables controls, adds the user message,
  // calls the unified `/chat` API, finishes progress, renders the assistant
  // answer, and finally re-enables controls. Every early return should happen
  // before `busy` is set, otherwise the UI could get stuck.
  event.preventDefault();
  if (state.busy) return;
  const question = els.questionInput.value.trim();
  if (!question) return;
  const remoteMedia = hasRemoteMediaUrl(question);
  const submittedAttachment = state.attachment;
  // The composer preview URL is released as soon as the request is submitted.
  // Give the transcript its own URL so the sent-image thumbnail remains visible.
  const submittedImagePreview = submittedAttachment?.file
    ? { name: submittedAttachment.name, url: URL.createObjectURL(submittedAttachment.file) }
    : null;
  const isMultimodalRequest = Boolean(submittedAttachment || remoteMedia);
  const askedAt = Date.now();
  const visibleConversationTitle = els.conversationList?.querySelector(".conversation-item.active .conversation-title");
  if (visibleConversationTitle && visibleConversationTitle.textContent === "新的咨询") {
    visibleConversationTitle.textContent = cleanQuestion(question).slice(0, 20) || "当前咨询";
  }

  // The submitted question is already preserved in the chat transcript. Clear
  // the composer immediately so it is ready for the next question.
  els.questionInput.value = "";
  autoResizeInput();
  renderQuestionMenu();
  // 点击发送后立即清空输入区；本次请求使用 submittedAttachment，不再依赖
  // 输入框当前状态，因此接口处理期间不会继续显示已发送图片。
  if (submittedAttachment) setAttachment(null);

  state.busy = true;
  els.sendBtn.disabled = true;
  els.uploadBtn.disabled = true;
  if (els.modelMenuBtn) els.modelMenuBtn.disabled = true;
  if (els.historyContextToggle) els.historyContextToggle.disabled = true;
  closeModelMenu();
  // Open a fresh RAG process record for this question and make the sidebar follow it.
  const proc = startProcess({
    question,
    requestKind: isMultimodalRequest ? "visual_manual" : null,
  });
  const userWrap = addMessage(
    "user",
    isMultimodalRequest ? userQuestionWithMedia(question, remoteMedia, submittedImagePreview) : question,
    { askedAt },
  );
  let questionTranslationPromise = null;
  if (isEnglishQuestion(question)) {
    questionTranslationPromise = prepareEnglishQuestionTranslation(question).catch((error) => {
      console.warn("Automatic English question translation failed", error);
      return "";
    });
    void questionTranslationPromise.then((translation) => {
      appendInlineQuestionTranslation(userWrap, question, translation);
    });
  }
  const loadingBubble = addMessage("assistant", loadingNode());
  bindProcessToMessage(userWrap, proc.id);
  bindProcessToMessage(loadingBubble, proc.id);
  try {
    await ensureAccountConversation(question);
    await saveAccountMessage("user", question, { timing: { askedAt } });
  } catch (historyError) {
    console.warn("保存用户咨询失败", historyError);
  }

  let multimodalStages = null;
    if (submittedAttachment && remoteMedia) {
    multimodalStages = [
      { pct: 0, text: "正在提交包含上传图片和图片链接的多模态请求..." },
      { pct: 14, text: "正在解析上传图片并安全下载链接图片..." },
      { pct: 36, text: "图片解析完成后，正在检索产品手册证据..." },
      { pct: 62, text: "智能体正在结合全部图片描述生成回复..." },
      { pct: 82, text: "正在整理答案文本和关联图片..." },
    ];
  } else if (submittedAttachment) {
    multimodalStages = [
        { pct: 0, text: "正在提交 /chat 多模态请求..." },
        { pct: 14, text: "正在解析上传图片..." },
        { pct: 36, text: "图片解析完成后，正在检索产品手册证据..." },
        { pct: 62, text: "智能体正在结合图片描述生成回复..." },
        { pct: 82, text: "正在整理答案文本和关联图片..." },
    ];
  } else if (remoteMedia) {
    multimodalStages = [
      { pct: 0, text: "检测到图片链接，正在提交多模态请求..." },
      { pct: 14, text: "正在安全下载并校验链接图片..." },
      { pct: 36, text: "链接图片解析完成后，正在检索产品手册证据..." },
      { pct: 62, text: "智能体正在结合图片内容生成回复..." },
      { pct: 82, text: "正在整理答案文本和关联图片..." },
    ];
  }
  const progress = createApiProgress(
    multimodalStages,
    isMultimodalRequest ? MIN_MULTIMODAL_PROGRESS_MS : MIN_PROGRESS_MS,
  );
  if (isMultimodalRequest) {
    const linkCount = (question.match(/https?:\/\/[^\s<>"']+/gi) || []).length;
    const mediaParts = [];
    if (submittedAttachment) mediaParts.push("上传图片 1 张");
    if (linkCount) mediaParts.push(`图片链接 ${linkCount} 个`);
    appendProgressTerminal(
      "vision",
      `读图：${mediaParts.join("、") || "图片输入"}；提取可见对象、文字与结构事实，供手册定位使用`,
      0,
    );
    setProgressStage("读图");
    setProgressStatus("正在读取图片并提取可见事实...");
  }
  const requestId = createId("kf_req");
  const progressLogs = startProgressLogPolling(requestId);
  let item;
  let englishTranslationPromise = null;
  let completedAt = null;
  let thinkingMs = null;
  try {
    let streamedText = "";
    const thinkingStartedAt = performance.now();
    item = await callRealApi(
      question,
      requestId,
      (delta) => {
        if (!delta) return;
        streamedText += delta;
        let liveAnswer = loadingBubble.querySelector(".streaming-answer");
        if (!liveAnswer) {
          loadingBubble.innerHTML = "";
          liveAnswer = document.createElement("div");
          liveAnswer.className = "answer-text streaming-answer";
          loadingBubble.appendChild(liveAnswer);
        }
        liveAnswer.textContent = streamedText;
        els.messages.scrollTop = els.messages.scrollHeight;
      },
      (status) => {
        const stage = String(status?.stage || "info");
        const message = String(status?.message || "").trim();
        setProgressStage(progressStageLabel(stage));
        if (message) setProgressStatus(message);
        appendProgressTerminal(stage, message, (performance.now() - thinkingStartedAt) / 1000);
      },
      (trace) => {
        if (!hasUsableAuditTrace(trace)) return;
        proc.audit = mergeAuditTrace(proc.audit, trace);
        liveTouch();
      },
      submittedAttachment,
    );
    syncActiveProductScope(item.contextProduct);
    // An English user keeps the vetted English answer as the primary display.
    // Translation runs independently while that answer is streaming, then the
    // Chinese copy is appended when ready so it never delays the first answer.
    if (isEnglishQuestion(question)) {
      englishTranslationPromise = prepareEnglishAnswerTranslation(item).catch((error) => {
        console.warn("Automatic English translation failed", error);
        return null;
      });
    }
    if (item.answerMode === "customer") {
      await simulateCustomerAnswerStream(loadingBubble, item.answer, thinkingStartedAt);
    } else if (!streamedText.trim()) {
      // Live manual RAG already emits delta events. Reviewed/table manual
      // answers intentionally do not, so stream their final vetted text here.
      await simulateManualAnswerStream(loadingBubble, item.answer, thinkingStartedAt);
    }
    completedAt = Date.now();
    thinkingMs = performance.now() - thinkingStartedAt;
  } catch (error) {
    progressLogs.stop();
    console.warn(error);
    await progress.finish("接口调用失败", "error");
    loadingBubble.remove();
    const errWrap = addMessage("assistant", errorNode(error));
    bindProcessToMessage(errWrap, proc.id);
    setActiveProcess(proc.id);
    state.busy = false;
    els.sendBtn.disabled = false;
    els.uploadBtn.disabled = false;
    if (els.modelMenuBtn) els.modelMenuBtn.disabled = false;
    if (els.historyContextToggle) els.historyContextToggle.disabled = false;
    return;
  }
  progressLogs.stop();
  const sourceLabel = item.source && item.source !== "api" ? `（${item.source}）` : "";
  await progress.finish(
    `API 调用成功${sourceLabel}`,
    "api",
  );

  loadingBubble.remove();
  const answerWrap = addMessage("assistant", renderAnswer(item), { completedAt, thinkingMs });
  bindProcessToMessage(answerWrap, proc.id);
  if (englishTranslationPromise) {
    const answerContent = answerWrap.querySelector(".answer-content");
    void englishTranslationPromise.then((translation) => replaceWithInlineChineseTranslation(answerContent, item, translation));
  }
  proc.images = item.images || [];
  // The final payload may carry a route-only trace. Never let that empty shell
  // erase the full audit received while the model was generating.
  if (hasUsableAuditTrace(item.retrievalTrace)) {
    proc.audit = mergeAuditTrace(proc.audit, item.retrievalTrace);
  }
  proc.summary = {
    mode: item.modeLabel || (item.answerMode === "customer" ? "客服模式" : "手册模式"),
    modeKey: item.answerMode === "customer" ? "customer" : "manual",
    imageCount: (item.images || []).length,
    manualCount: (item.manuals || []).length,
    elapsed: `${(Number(proc.elapsed) || 0).toFixed(1)}s`,
  };
  proc.contextTurns = Number(item.contextTurns || 0);
  setActiveProcess(proc.id);
  rememberConversationTurn(question, item);
  try {
    await saveAccountMessage(
      "assistant",
      item.answer || "",
      compactHistoryPayload(item, { askedAt, completedAt, thinkingMs }),
    );
    await loadConversationList();
  } catch (historyError) {
    console.warn("保存客服回答失败", historyError);
  }
  state.busy = false;
  els.sendBtn.disabled = false;
  els.uploadBtn.disabled = false;
  if (els.modelMenuBtn) els.modelMenuBtn.disabled = false;
  if (els.historyContextToggle) els.historyContextToggle.disabled = false;
}

function setProgressStatus(text, kind = null) {
  // Write into the live record; the sidebar repaints if it is the active view.
  if (state.live) {
    state.live.status = text || "";
    state.live.kind = kind;
  }
  liveTouch();
}

function userQuestionWithMedia(question, remoteMedia = false, attachment = null) {
  // Render the submitted image in the user bubble. The data URL sent upstream
  // remains private; this is a browser-local object URL created just for display.
  const wrap = document.createElement("div");
  const text = document.createElement("div");
  text.textContent = question;
  wrap.append(text);
  if (attachment?.url) {
    const preview = document.createElement("img");
    preview.className = "user-sent-image";
    preview.src = attachment.url;
    preview.alt = attachment.name ? `已发送图片：${attachment.name}` : "已发送图片";
    preview.addEventListener("load", () => URL.revokeObjectURL(attachment.url), { once: true });
    preview.addEventListener("error", () => URL.revokeObjectURL(attachment.url), { once: true });
    wrap.append(preview);
    const attachmentTag = document.createElement("div");
    attachmentTag.className = "user-attachment-tag";
    attachmentTag.textContent = `已附图：${attachment.name || "图片"}`;
    wrap.append(attachmentTag);
  }
  if (remoteMedia) {
    const linkTag = document.createElement("div");
    linkTag.className = "user-attachment-tag";
    linkTag.textContent = "已检测图片链接，将自动读取";
    wrap.append(linkTag);
  }
  return wrap;
}

function loadingNode() {
  // Initial assistant placeholder shown while the real/table path is working.
  const div = document.createElement("div");
  div.className = "status-line";
  div.innerHTML = '<span class="dot"></span><span>正在调用智能体检索产品手册并生成回复...</span>';
  return div;
}

function autoResizeInput() {
  // Auto-grow the textarea up to a fixed maximum so long questions remain
  // readable without pushing the whole page out of the fixed-height layout.
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(140, els.questionInput.scrollHeight)}px`;
}

function setAttachment(file) {
  // Manage one optional image attachment. Object URLs are revoked when replaced
  // or removed so repeated uploads do not leak browser memory during demos.
  if (state.attachment?.url) URL.revokeObjectURL(state.attachment.url);
  if (!file) {
    state.attachment = null;
    els.attachmentPreview.hidden = true;
    els.attachmentPreview.innerHTML = "";
    els.imageInput.value = "";
    return;
  }
  const url = URL.createObjectURL(file);
  state.attachment = { file, name: file.name, url };
  els.attachmentPreview.hidden = false;
  els.attachmentPreview.innerHTML = `
    <div class="attachment-thumb"><img src="${url}" alt=""></div>
    <div class="attachment-meta">
      <div class="attachment-name">${escapeHtml(file.name)}</div>
      <div class="attachment-note">图片已加入本轮提问；接口会按 /chat 的 images 字段提交。</div>
    </div>
    <button class="attachment-remove" type="button" aria-label="移除图片">×</button>
  `;
  els.attachmentPreview.querySelector(".attachment-remove").addEventListener("click", () => setAttachment(null));
}

function clipboardImageFile(event) {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.kind === "file" && item.type.startsWith("image/"));
  if (!imageItem) return null;
  const image = imageItem.getAsFile();
  if (!image) return null;
  if (image.name) return image;
  const extension = (image.type.split("/")[1] || "png").replace(/[^a-z0-9]/gi, "") || "png";
  return new File([image], `clipboard-image-${Date.now()}.${extension}`, { type: image.type });
}

function readFileAsDataUrl(file) {
  // Convert the selected image to the exact data-url format expected by `/chat`.
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Image read failed"));
    reader.readAsDataURL(file);
  });
}

function createId(prefix) {
  // Prefer crypto-quality UUIDs when available. The fallback is good enough for
  // local request/session labels and keeps older browser engines working.
  if (window.crypto?.randomUUID) {
    return `${prefix}_${window.crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function escapeHtml(value) {
  // All product names and filenames are data-driven, so escape before inserting
  // into HTML strings. This keeps the no-framework rendering approach safe.
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadModelProfiles() {
  if (!els.modelMenuBtn) return;
  try {
    const res = await fetch(MODEL_PROFILE_ENDPOINT, { cache: "no-store" });
    const payload = await res.json();
    if (!res.ok || payload.code !== 0) throw new Error(payload.msg || "model profile load failed");
    state.modelProfiles = payload.data?.profiles || [];
    state.activeModelProfile = payload.data?.active || state.modelProfiles[0] || null;
    state.reasoningEffort = "medium";
    renderModelSwitcher();
  } catch (error) {
    console.warn(error);
    if (els.activeModelLabel) els.activeModelLabel.textContent = "模型配置不可用";
    els.modelMenuBtn.classList.add("is-error");
  }
}

function renderModelSwitcher() {
  const active = state.activeModelProfile;
  const iconName = active?.icon || "gpt";
  if (els.activeModelLabel) {
    els.activeModelLabel.textContent = active?.label || "选择模型";
  }
  if (els.activeModelIcon) {
    els.activeModelIcon.src = `./model-${iconName}.svg?v=model-icons-20260730c`;
    els.activeModelIcon.alt = active?.provider || "GPT";
  }
  if (!els.modelMenu) return;
  const profiles = state.modelProfiles || [];
  els.modelMenu.innerHTML = `
    <div class="model-menu-header">
      <div class="model-menu-title">选择模型</div>
      <div class="model-menu-subtitle">切换会影响后续回答，当前请求不会被中断。</div>
    </div>
    <div class="model-option-list">
      ${profiles.map((profile) => modelOptionTemplate(profile, active?.id === profile.id)).join("")}
    </div>
  `;
  els.modelMenu.querySelectorAll(".model-option").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      switchModelProfile(button.dataset.profileId);
    });
  });
  renderReasoningSwitcher();
}

const REASONING_LABELS = { low: "低", medium: "中", high: "高" };

function reasoningOptionsFor(profile = state.activeModelProfile) {
  return Array.isArray(profile?.reasoning_options) ? profile.reasoning_options : ["medium"];
}

function renderReasoningSwitcher() {
  if (!els.reasoningSwitcher || !els.reasoningMenu) return;
  // Product policy: keep reasoning at a predictable medium level and do not
  // expose this internal tuning control in the customer-facing composer.
  state.reasoningEffort = "medium";
  els.reasoningSwitcher.hidden = true;
  closeReasoningMenu();
}

function closeReasoningMenu() {
  state.reasoningMenuOpen = false;
  if (els.reasoningMenu) els.reasoningMenu.hidden = true;
  if (els.reasoningMenuBtn) els.reasoningMenuBtn.setAttribute("aria-expanded", "false");
}

function toggleReasoningMenu() {
  if (!els.reasoningMenu || !els.reasoningMenuBtn || !reasoningOptionsFor().length) return;
  state.reasoningMenuOpen = !state.reasoningMenuOpen;
  els.reasoningMenu.hidden = !state.reasoningMenuOpen;
  els.reasoningMenuBtn.setAttribute("aria-expanded", String(state.reasoningMenuOpen));
}

function modelOptionTemplate(profile, active) {
  const iconName = profile.icon || "gpt";
  const isUnavailable = profile.available === false;
  const badge = profile.id?.startsWith("deepseek-") ? "测试" : "默认";
  return `
    <button class="model-option${active ? " active" : ""}${isUnavailable ? " unavailable" : ""}" type="button" role="option" aria-selected="${active ? "true" : "false"}" data-profile-id="${escapeHtml(profile.id)}" ${isUnavailable ? "disabled" : ""}>
      <span class="model-option-check" aria-hidden="true">${active ? "✓" : ""}</span>
      <span class="model-option-main">
        <span class="model-option-row">
          <img class="model-provider-icon" src="./model-${escapeHtml(iconName)}.svg?v=model-icons-20260730c" alt="" aria-hidden="true">
          <span class="model-option-name">${escapeHtml(profile.label || profile.id)}</span>
          <span class="model-option-badge">${escapeHtml(profile.available === false ? "待接入" : (profile.text_only ? "文本" : "多模态"))}</span>
        </span>
        <span class="model-option-desc">${escapeHtml(profile.description || "")}</span>
        <span class="model-option-meta">${escapeHtml(profile.provider || "")} · ${escapeHtml(profile.model || "")}</span>
      </span>
    </button>
  `;
}

function openModelMenu() {
  if (!els.modelMenu || !els.modelMenuBtn || state.busy) return;
  state.modelMenuOpen = true;
  els.modelMenu.hidden = false;
  els.modelMenuBtn.setAttribute("aria-expanded", "true");
}

function closeModelMenu() {
  if (!els.modelMenu || !els.modelMenuBtn) return;
  state.modelMenuOpen = false;
  els.modelMenu.hidden = true;
  els.modelMenuBtn.setAttribute("aria-expanded", "false");
}

function toggleModelMenu() {
  if (state.modelMenuOpen) closeModelMenu();
  else openModelMenu();
}

async function switchModelProfile(profileId) {
  if (!profileId || state.busy) return;
  const previous = state.activeModelProfile;
  const next = state.modelProfiles.find((profile) => profile.id === profileId);
  if (!next || previous?.id === next.id) {
    closeModelMenu();
    return;
  }
  els.modelMenuBtn.disabled = true;
  if (els.activeModelLabel) els.activeModelLabel.textContent = "正在切换...";
  try {
    const res = await fetch(MODEL_PROFILE_SWITCH_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${CHAT_API_TOKEN}`,
      },
      body: JSON.stringify({ profile_id: profileId }),
    });
    const payload = await res.json();
    if (!res.ok || payload.code !== 0) throw new Error(payload.msg || "模型切换失败");
    state.modelProfiles = payload.data?.profiles || state.modelProfiles;
    state.activeModelProfile = payload.data?.active || next;
    state.reasoningEffort = "medium";
    renderModelSwitcher();
    closeModelMenu();
  } catch (error) {
    console.warn(error);
    state.activeModelProfile = previous;
    renderModelSwitcher();
    setProgressStatus(`模型切换失败：${error.message || error}`);
  } finally {
    els.modelMenuBtn.disabled = false;
  }
}

async function accountApi(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.code !== 0) throw new Error(payload.msg || payload.error || `HTTP ${response.status}`);
  return payload.data || {};
}

function showAuthModal(mode = "login") {
  setAuthFormMode(mode);
  els.authError.textContent = "";
  els.authModal.hidden = false;
  window.setTimeout(() => els.authUsername.focus(), 20);
}

function hideAuthModal() {
  els.authModal.hidden = true;
  els.authError.textContent = "";
}

function setAuthFormMode(mode) {
  state.authFormMode = mode === "register" ? "register" : "login";
  els.loginTab.classList.toggle("active", state.authFormMode === "login");
  els.registerTab.classList.toggle("active", state.authFormMode === "register");
  els.authSubmit.textContent = state.authFormMode === "login" ? "登录并继续" : "注册并继续";
  els.authPassword.autocomplete = state.authFormMode === "login" ? "current-password" : "new-password";
  els.authError.textContent = "";
}

function closeAccountMenu() {
  els.accountMenu.hidden = true;
  els.accountButton.setAttribute("aria-expanded", "false");
}

function renderAccountState() {
  const user = state.authUser;
  if (user) {
    const name = user.display_name || user.username;
    els.accountAvatar.textContent = name.slice(0, 1).toUpperCase();
    els.accountName.textContent = name;
    els.accountHint.textContent = "历史咨询已同步";
    els.accountMenuIdentity.textContent = `已登录：${user.username}`;
    els.accountLoginAction.hidden = true;
    els.accountLogoutAction.hidden = false;
    els.conversationNote.textContent = state.conversations.length ? "点击历史咨询可恢复查看" : "首次提问后将自动保存咨询记录";
  } else {
    els.accountAvatar.textContent = "访";
    els.accountName.textContent = "游客模式";
    els.accountHint.textContent = "历史按网络地址保存";
    els.accountMenuIdentity.textContent = "游客模式：历史按当前网络地址安全保存";
    els.accountLoginAction.hidden = false;
    els.accountLogoutAction.hidden = true;
    els.conversationNote.textContent = state.conversations.length
      ? "点击历史咨询可恢复查看；更换网络后记录可能变化"
      : "首次提问后将按当前网络地址自动保存";
  }
}

function conversationTime(timestamp) {
  if (!timestamp) return "";
  const value = new Date(Number(timestamp));
  const today = new Date();
  if (value.toDateString() === today.toDateString()) {
    return value.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  }
  return `${value.getMonth() + 1}/${value.getDate()}`;
}

function renderConversationList() {
  if (!els.conversationList) return;
  const rows = [];
  if (!state.activeConversationId) {
    rows.push(`
      <button class="conversation-item active" type="button" data-conversation-new="1">
        <span class="conversation-icon" aria-hidden="true">◇</span>
        <span class="conversation-title">新的咨询</span>
        <span class="conversation-status">当前</span>
      </button>
    `);
  }
  for (const conversation of state.conversations) {
    const active = conversation.id === state.activeConversationId;
    rows.push(`
      <button class="conversation-item history${active ? " active" : ""}" type="button" data-conversation-id="${escapeHtml(conversation.id)}">
        <span class="conversation-icon" aria-hidden="true">◇</span>
        <span class="conversation-title">${escapeHtml(conversation.title || "新的咨询")}</span>
        <span class="conversation-status">${active ? "当前" : escapeHtml(conversationTime(conversation.updated_at))}</span>
      </button>
    `);
  }
  els.conversationList.innerHTML = rows.join("");
  renderAccountState();
}

async function loadConversationList() {
  try {
    const data = await accountApi(CONVERSATIONS_ENDPOINT);
    state.conversations = data.conversations || [];
  } catch (error) {
    console.warn("load conversations failed", error);
    state.conversations = [];
  }
  renderConversationList();
}

function emptyConversationCanvas() {
  els.messages.innerHTML = `
    <div class="welcome">
      <img class="welcome-logo" src="/rag/ray-source-mark.png?v=r-only-20260728" alt="睿视清源 R" width="96" height="96">
      <div class="welcome-title">有什么可以帮忙的？</div>
      <p>支持上传图片或粘贴图片链接；我会结合产品手册回答并展示相关图片。</p>
    </div>`;
  state.processes.clear();
  state.live = null;
  renderIdleProcess();
}

function rebuildStructuredMemory(messages) {
  state.productMemories.clear();
  let pendingQuestion = "";
  let lastProduct = "";
  for (const message of messages || []) {
    if (message.role === "user") {
      pendingQuestion = message.content || "";
      continue;
    }
    if (message.role !== "assistant" || !pendingQuestion) continue;
    const item = message.payload || {};
    const product = item.contextProduct || item.product || "";
    if (!product || !knownProductNames().includes(product)) {
      pendingQuestion = "";
      continue;
    }
    const turns = memoryTurnsFor(product);
    turns.push({
      question: pendingQuestion,
      answer: item.answer || message.content || "",
      product,
      modeLabel: item.modeLabel || (item.answerMode === "customer" ? "客服模式" : "手册模式"),
      imageDescriptions: item.imageDescriptions || [],
    });
    state.productMemories.set(product, turns.slice(-MAX_HISTORY_TURNS));
    lastProduct = product;
    pendingQuestion = "";
  }
  if (lastProduct) syncActiveProductScope(lastProduct);
  else updateHistoryContextIndicator();
  persistProductMemories();
}

async function loadSavedConversation(conversationId) {
  if (state.busy) return;
  try {
    const data = await accountApi(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(conversationId)}`);
    state.activeConversationId = data.conversation.id;
    els.messages.innerHTML = "";
    for (const message of data.messages || []) {
      if (message.role === "user") {
        addMessage("user", message.content, {
          askedAt: message.payload?.timing?.askedAt || message.created_at,
        });
      } else {
        const item = message.payload || {
          answer: message.content,
          product: "产品",
          images: [],
          manuals: [],
          sources: [],
          imageDescriptions: [],
          answerMode: "manual",
          modeLabel: "历史回答",
        };
        addMessage("assistant", renderAnswer(item), {
          completedAt: message.payload?.timing?.completedAt || message.created_at,
          thinkingMs: message.payload?.timing?.thinkingMs,
        });
      }
    }
    rebuildStructuredMemory(data.messages || []);
    if (!(data.messages || []).length) emptyConversationCanvas();
    state.processes.clear();
    state.live = null;
    renderIdleProcess();
    renderConversationList();
  } catch (error) {
    console.warn(error);
    setProgressStatus(`历史咨询加载失败：${error.message}`, "error");
  }
}

async function ensureAccountConversation(question) {
  if (state.activeConversationId) return state.activeConversationId;
  const title = cleanQuestion(question).slice(0, 36) || "新的咨询";
  const data = await accountApi(CONVERSATIONS_ENDPOINT, {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  state.activeConversationId = data.conversation.id;
  state.conversations.unshift({ ...data.conversation, message_count: 0 });
  renderConversationList();
  return state.activeConversationId;
}

function compactHistoryPayload(item, timing = null) {
  return {
    requestId: item.requestId || "",
    question: item.question || "",
    answer: item.answer || "",
    product: item.product || "",
    images: item.images || [],
    manuals: item.manuals || [],
    sources: item.sources || [],
    imageDescriptions: item.imageDescriptions || [],
    answerMode: item.answerMode || "manual",
    modeLabel: item.modeLabel || "手册模式",
    routing: item.routing || {},
    contextPacketVersion: item.contextPacketVersion || 0,
    contextRetrievalHint: item.contextRetrievalHint || "auto",
    timing: timing || null,
  };
}

async function saveAccountMessage(role, content, payload = null) {
  if (!state.activeConversationId) return;
  await accountApi(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(state.activeConversationId)}/messages`, {
    method: "POST",
    body: JSON.stringify({ role, content, payload }),
  });
}

async function refreshAccountState({ promptIfGuest = false } = {}) {
  try {
    const data = await accountApi(ACCOUNT_ME_ENDPOINT);
    state.authUser = data.user || null;
    state.authMode = state.authUser ? "account" : "guest";
  } catch {
    state.authUser = null;
    state.authMode = "guest";
  }
  await loadConversationList();
  renderAccountState();
  const guestChosen = window.localStorage.getItem("ragv6_guest_mode") === "1";
  if (state.authUser) {
    window.localStorage.removeItem("ragv6_guest_mode");
    hideAuthModal();
  } else if (promptIfGuest && !guestChosen) {
    showAuthModal("login");
  }
}

async function submitAuthForm(event) {
  event.preventDefault();
  els.authError.textContent = "";
  els.authSubmit.disabled = true;
  try {
    const endpoint = state.authFormMode === "register" ? ACCOUNT_REGISTER_ENDPOINT : ACCOUNT_LOGIN_ENDPOINT;
    const data = await accountApi(endpoint, {
      method: "POST",
      body: JSON.stringify({
        username: els.authUsername.value.trim(),
        password: els.authPassword.value,
      }),
    });
    state.authUser = data.user;
    state.authMode = "account";
    state.activeConversationId = null;
    window.localStorage.removeItem("ragv6_guest_mode");
    els.authForm.reset();
    hideAuthModal();
    await loadConversationList();
    renderAccountState();
  } catch (error) {
    els.authError.textContent = error.message;
  } finally {
    els.authSubmit.disabled = false;
  }
}

async function continueAsGuest() {
  state.authUser = null;
  state.authMode = "guest";
  state.activeConversationId = null;
  window.localStorage.setItem("ragv6_guest_mode", "1");
  hideAuthModal();
  await loadConversationList();
  renderConversationList();
  renderAccountState();
}

async function logoutAccount() {
  try {
    await accountApi(ACCOUNT_LOGOUT_ENDPOINT, { method: "POST", body: "{}" });
  } catch (error) {
    console.warn(error);
  }
  state.authUser = null;
  state.authMode = "guest";
  state.activeConversationId = null;
  window.localStorage.setItem("ragv6_guest_mode", "1");
  closeAccountMenu();
  emptyConversationCanvas();
  await loadConversationList();
  renderConversationList();
  renderAccountState();
}

async function init() {
  // Load demo data without cache so edits to `answers.json` show up immediately
  // after refresh. The first product is selected to make the page useful without
  // any extra clicks.
  renderIdleProcess();
  loadModelProfiles();
  const res = await fetch("./answers.json", { cache: "no-store" });
  state.data = await res.json();
  renderProducts();
  selectProduct(state.data.products[0].name);
  await refreshAccountState({ promptIfGuest: true });
}

// Event wiring is intentionally collected at the bottom: after reading the
// functions above, maintainers can see all user interactions in one place.
function startNewChat() {
  // Pure UI reset: clear the transcript and rolling history, restore the welcome
  // splash. Does not touch any backend state or session contract.
  if (state.busy) return;
  state.productMemories.clear();
  state.productMemoryEpochs.clear();
  state.sessionId = null;
  window.localStorage.removeItem("ragv6_session_id");
  window.localStorage.removeItem(MEMORY_EPOCH_STORAGE_KEY);
  window.localStorage.removeItem(PRODUCT_MEMORY_STORAGE_KEY);
  state.activeConversationId = null;
  updateHistoryContextIndicator();
  emptyConversationCanvas();
  renderConversationList();
  closeMobileDrawers();
  els.questionInput.focus();
}

function toggleLogbar() {
  if (!els.appShell) return;
  els.appShell.classList.toggle("logbar-collapsed");
}

function isPhoneLayout() {
  return window.matchMedia("(max-width: 820px)").matches;
}

function mobileLogbarViewport() {
  return Math.max(280, Math.round(window.visualViewport?.width || window.innerWidth || 360));
}

function mobileLogbarBounds() {
  const viewport = mobileLogbarViewport();
  const min = Math.min(320, Math.max(248, Math.round(viewport * 0.68)));
  const max = viewport;
  const preferred = Math.min(max, Math.max(min, Math.round(viewport * 0.88)));
  return { min, max, preferred };
}

function clampMobileLogbarWidth(width) {
  const bounds = mobileLogbarBounds();
  const value = Number(width);
  if (!Number.isFinite(value)) return bounds.preferred;
  return Math.round(Math.min(bounds.max, Math.max(bounds.min, value)));
}

function setMobileLogbarWidth(width, { persist = true } = {}) {
  if (!els.logbar) return;
  const value = clampMobileLogbarWidth(width);
  els.logbar.style.setProperty("--mobile-logbar-width", `${value}px`);
  if (els.mobileLogbarResize) {
    const bounds = mobileLogbarBounds();
    els.mobileLogbarResize.setAttribute("aria-valuemin", String(bounds.min));
    els.mobileLogbarResize.setAttribute("aria-valuemax", String(bounds.max));
    els.mobileLogbarResize.setAttribute("aria-valuenow", String(value));
  }
  if (persist) {
    try {
      window.localStorage.setItem(MOBILE_LOGBAR_WIDTH_STORAGE_KEY, String(value));
    } catch {
      // Private browsing may make localStorage unavailable; the current width
      // remains usable for this session.
    }
  }
}

function restoreMobileLogbarWidth() {
  let stored = null;
  try {
    stored = window.localStorage.getItem(MOBILE_LOGBAR_WIDTH_STORAGE_KEY);
  } catch {
    stored = null;
  }
  setMobileLogbarWidth(stored || mobileLogbarBounds().preferred, { persist: false });
}

function setMobileLogbarExpanded(expanded) {
  if (!els.appShell) return;
  els.appShell.classList.toggle("mobile-logbar-expanded", !!expanded);
  if (els.mobileLogbarExpand) {
    els.mobileLogbarExpand.setAttribute("aria-pressed", String(!!expanded));
    els.mobileLogbarExpand.setAttribute("aria-label", expanded ? "恢复思考栏宽度" : "展开思考栏");
    els.mobileLogbarExpand.title = expanded ? "恢复思考栏宽度" : "展开思考栏";
  }
}

function syncMobileDrawers() {
  if (!els.appShell) return;
  const sidebarOpen = els.appShell.classList.contains("mobile-sidebar-open");
  const logbarOpen = els.appShell.classList.contains("mobile-logbar-open");
  if (els.mobileSidebarToggle) {
    els.mobileSidebarToggle.setAttribute("aria-expanded", String(sidebarOpen));
  }
  if (els.mobileLogbarToggle) {
    els.mobileLogbarToggle.setAttribute("aria-expanded", String(logbarOpen));
  }
  if (els.mobileDrawerBackdrop) {
    els.mobileDrawerBackdrop.hidden = !(sidebarOpen || logbarOpen);
  }
  if (els.logbar && isPhoneLayout()) {
    els.logbar.setAttribute("aria-hidden", String(!logbarOpen));
  } else if (els.logbar) {
    els.logbar.removeAttribute("aria-hidden");
  }
}

function openMobileLogbar() {
  if (!els.appShell) return;
  restoreMobileLogbarWidth();
  setMobileLogbarExpanded(false);
  els.appShell.classList.add("mobile-logbar-open");
  els.appShell.classList.remove("mobile-sidebar-open");
  syncMobileDrawers();
}

function closeMobileDrawers() {
  if (!els.appShell) return;
  els.appShell.classList.remove("mobile-sidebar-open", "mobile-logbar-open", "mobile-logbar-expanded");
  if (els.mobileLogbarExpand) els.mobileLogbarExpand.setAttribute("aria-pressed", "false");
  syncMobileDrawers();
}

function toggleMobileSidebar() {
  if (!els.appShell) return;
  const shouldOpen = !els.appShell.classList.contains("mobile-sidebar-open");
  els.appShell.classList.toggle("mobile-sidebar-open", shouldOpen);
  els.appShell.classList.remove("mobile-logbar-open", "mobile-logbar-expanded");
  syncMobileDrawers();
}

function toggleMobileLogbar() {
  if (!els.appShell) return;
  const shouldOpen = !els.appShell.classList.contains("mobile-logbar-open");
  if (shouldOpen) openMobileLogbar();
  else closeMobileDrawers();
}

function mobileInputId(event) {
  return event.pointerId ?? event.touches?.[0]?.identifier ?? event.changedTouches?.[0]?.identifier;
}

function mobileInputX(event) {
  return event.clientX ?? event.touches?.[0]?.clientX ?? event.changedTouches?.[0]?.clientX;
}

function mobileInputY(event) {
  return event.clientY ?? event.touches?.[0]?.clientY ?? event.changedTouches?.[0]?.clientY;
}

function beginMobileLogbarResize(event) {
  if (!isPhoneLayout() || !els.appShell?.classList.contains("mobile-logbar-open")) return;
  const pointerId = mobileInputId(event);
  if (pointerId == null) return;
  event.preventDefault();
  event.stopPropagation();
  setMobileLogbarExpanded(false);
  mobileLogbarResizeState = {
    pointerId,
    target: event.currentTarget,
    usesPointerCapture: "pointerId" in event,
  };
  if (mobileLogbarResizeState.usesPointerCapture) {
    try {
      mobileLogbarResizeState.target.setPointerCapture?.(pointerId);
    } catch {
      // Android WebView can reject capture while a transformed drawer is opening.
      // The window-level listeners below still retain the resize gesture.
    }
  }
}

function moveMobileLogbarResize(event) {
  if (!mobileLogbarResizeState || mobileInputId(event) !== mobileLogbarResizeState.pointerId) return;
  const clientX = mobileInputX(event);
  if (!Number.isFinite(clientX)) return;
  event.preventDefault();
  setMobileLogbarWidth(mobileLogbarViewport() - clientX, { persist: false });
}

function endMobileLogbarResize(event) {
  if (!mobileLogbarResizeState || mobileInputId(event) !== mobileLogbarResizeState.pointerId) return;
  const target = mobileLogbarResizeState.target;
  const usesPointerCapture = mobileLogbarResizeState.usesPointerCapture;
  mobileLogbarResizeState = null;
  if (usesPointerCapture) {
    try {
      target.releasePointerCapture?.(event.pointerId);
    } catch {
      // The pointer may already have been released by the browser.
    }
  }
  const current = els.logbar?.getBoundingClientRect().width;
  if (Number.isFinite(current)) setMobileLogbarWidth(current);
}

function adjustMobileLogbarWidth(event) {
  if (!isPhoneLayout()) return;
  const current = els.logbar?.getBoundingClientRect().width || mobileLogbarBounds().preferred;
  const bounds = mobileLogbarBounds();
  let next = current;
  if (event.key === "ArrowLeft") next += 24;
  else if (event.key === "ArrowRight") next -= 24;
  else if (event.key === "Home") next = bounds.max;
  else if (event.key === "End") next = bounds.min;
  else return;
  event.preventDefault();
  setMobileLogbarExpanded(false);
  setMobileLogbarWidth(next);
}

function beginMobileDrawerSwipe(event) {
  if (!isPhoneLayout() || mobileLogbarResizeState) return;
  const pointerId = mobileInputId(event);
  const x = Number(mobileInputX(event));
  const y = Number(mobileInputY(event));
  if (pointerId == null || !Number.isFinite(x) || !Number.isFinite(y)) return;
  const viewport = mobileLogbarViewport();
  const logbarOpen = els.appShell?.classList.contains("mobile-logbar-open");
  // Android reserves the physical screen edge for Back. Keep the swipe start
  // zone inside the page so the audit drawer remains reachable on gesture-nav.
  if (!logbarOpen && x < viewport - 48) return;
  if (logbarOpen && !els.logbar?.contains(event.target)) return;
  mobileLogbarSwipeState = {
    pointerId,
    inputKind: event.type.startsWith("touch") ? "touch" : "pointer",
    startX: x,
    startY: y,
    opening: !logbarOpen,
  };
}

function endMobileDrawerSwipe(event) {
  if (!mobileLogbarSwipeState || mobileInputId(event) !== mobileLogbarSwipeState.pointerId) return;
  const stateAtStart = mobileLogbarSwipeState;
  const inputKind = event.type.startsWith("touch") ? "touch" : "pointer";
  if (stateAtStart.inputKind !== inputKind) return;
  mobileLogbarSwipeState = null;
  const x = Number(mobileInputX(event));
  const y = Number(mobileInputY(event));
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const dx = x - stateAtStart.startX;
  const dy = y - stateAtStart.startY;
  if (Math.abs(dx) < 52 || Math.abs(dx) < Math.abs(dy) * 1.15) return;
  if (stateAtStart.opening && dx < 0) openMobileLogbar();
  if (!stateAtStart.opening && dx > 0) closeMobileDrawers();
}

if (els.newChatBtn) els.newChatBtn.addEventListener("click", startNewChat);
if (els.conversationList) {
  els.conversationList.addEventListener("click", (event) => {
    const button = event.target.closest(".conversation-item");
    if (!button) return;
    if (button.dataset.conversationId) loadSavedConversation(button.dataset.conversationId);
    else els.messages.scrollTo({ top: 0, behavior: "smooth" });
    if (window.matchMedia("(max-width: 820px)").matches) closeMobileDrawers();
  });
}
if (els.productSearch) {
  els.productSearch.addEventListener("input", () => renderProducts(els.productSearch.value));
}
if (els.historyContextToggle) {
  els.historyContextToggle.addEventListener("change", () => updateHistoryContextIndicator());
}
if (els.clearHistoryContext) {
  els.clearHistoryContext.addEventListener("click", () => {
    if (!state.activeProduct || state.busy) return;
    state.productMemories.delete(state.activeProduct);
    persistProductMemories();
    state.productMemoryEpochs.set(state.activeProduct, createId("memory_scope"));
    persistProductMemoryEpochs();
    updateHistoryContextIndicator();
  });
}
if (els.logbarToggle) els.logbarToggle.addEventListener("click", toggleLogbar);
if (els.mobileSidebarToggle) els.mobileSidebarToggle.addEventListener("click", toggleMobileSidebar);
if (els.mobileLogbarToggle) els.mobileLogbarToggle.addEventListener("click", toggleMobileLogbar);
if (els.mobileLogbarClose) els.mobileLogbarClose.addEventListener("click", closeMobileDrawers);
if (els.mobileLogbarExpand) els.mobileLogbarExpand.addEventListener("click", () => {
  if (!els.appShell?.classList.contains("mobile-logbar-open")) return;
  setMobileLogbarExpanded(!els.appShell.classList.contains("mobile-logbar-expanded"));
});
if (els.mobileLogbarResize) {
  if (window.PointerEvent) {
    els.mobileLogbarResize.addEventListener("pointerdown", beginMobileLogbarResize);
    // Track the resize gesture after the finger leaves the narrow handle.
    // Android WebView does not always retain pointer capture across fixed transforms.
    window.addEventListener("pointermove", moveMobileLogbarResize, { passive: false });
    window.addEventListener("pointerup", endMobileLogbarResize, { passive: false });
    window.addEventListener("pointercancel", endMobileLogbarResize, { passive: false });
  } else {
    els.mobileLogbarResize.addEventListener("touchstart", beginMobileLogbarResize, { passive: false });
    window.addEventListener("touchmove", moveMobileLogbarResize, { passive: false });
    window.addEventListener("touchend", endMobileLogbarResize, { passive: false });
    window.addEventListener("touchcancel", endMobileLogbarResize, { passive: false });
  }
  els.mobileLogbarResize.addEventListener("keydown", adjustMobileLogbarWidth);
}
if (window.PointerEvent) {
  document.addEventListener("pointerdown", beginMobileDrawerSwipe, { passive: true });
  window.addEventListener("pointerup", endMobileDrawerSwipe, { passive: true });
  window.addEventListener("pointercancel", endMobileDrawerSwipe, { passive: true });
}
// Some WebViews expose PointerEvent but dispatch synthetic/system gestures only
// as TouchEvent. Let a touch event take ownership of the same gesture.
document.addEventListener("touchstart", beginMobileDrawerSwipe, { passive: true });
window.addEventListener("touchend", endMobileDrawerSwipe, { passive: true });
window.addEventListener("touchcancel", endMobileDrawerSwipe, { passive: true });
if (els.mobileDrawerBackdrop) els.mobileDrawerBackdrop.addEventListener("click", closeMobileDrawers);
if (els.accountButton) {
  els.accountButton.addEventListener("click", (event) => {
    event.stopPropagation();
    els.accountMenu.hidden = !els.accountMenu.hidden;
    els.accountButton.setAttribute("aria-expanded", String(!els.accountMenu.hidden));
  });
}
if (els.accountLoginAction) {
  els.accountLoginAction.addEventListener("click", () => {
    closeAccountMenu();
    showAuthModal("login");
  });
}
if (els.accountLogoutAction) els.accountLogoutAction.addEventListener("click", logoutAccount);
if (els.loginTab) els.loginTab.addEventListener("click", () => setAuthFormMode("login"));
if (els.registerTab) els.registerTab.addEventListener("click", () => setAuthFormMode("register"));
if (els.authForm) els.authForm.addEventListener("submit", submitAuthForm);
if (els.guestContinue) els.guestContinue.addEventListener("click", continueAsGuest);
if (els.authClose) els.authClose.addEventListener("click", continueAsGuest);
if (els.authModal) els.authModal.querySelector(".auth-backdrop")?.addEventListener("click", continueAsGuest);
if (els.modelMenuBtn) {
  els.modelMenuBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleModelMenu();
  });
}
if (els.reasoningMenuBtn) {
  els.reasoningMenuBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleReasoningMenu();
  });
}
els.manualIndexBtn?.addEventListener("click", () => {
  window.location.assign("/rag/manual-index/");
});
els.mobileManualIndexBtn?.addEventListener("click", () => {
  closeMobileDrawers();
  window.location.assign("/rag/manual-index/");
});
els.uploadBtn.addEventListener("click", () => els.imageInput.click());
els.questionMenuBtn.addEventListener("click", toggleQuestionMenu);
els.questionField.addEventListener("mouseenter", cancelQuestionMenuClose);
els.questionField.addEventListener("mouseleave", scheduleQuestionMenuClose);
els.imageInput.addEventListener("change", () => setAttachment(els.imageInput.files?.[0] || null));
els.composer.addEventListener("submit", handleSubmit);
els.questionInput.addEventListener("input", () => {
  autoResizeInput();
  renderQuestionMenu();
});
els.questionInput.addEventListener("paste", (event) => {
  if (state.busy) return;
  const image = clipboardImageFile(event);
  if (!image) return;
  event.preventDefault();
  setAttachment(image);
});
els.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  } else if (event.key === "Escape") {
    closeQuestionMenu();
  }
});
document.addEventListener("click", (event) => {
  if (els.accountMenu && !els.accountMenu.hidden && !els.accountMenu.contains(event.target)) {
    closeAccountMenu();
  }
  if (els.modelMenu && !els.modelMenu.hidden) {
    if (event.target !== els.modelMenuBtn && !els.modelMenuBtn?.contains(event.target) && !els.modelMenu.contains(event.target)) {
      closeModelMenu();
    }
  }
  if (els.reasoningMenu && !els.reasoningMenu.hidden) {
    if (event.target !== els.reasoningMenuBtn && !els.reasoningMenuBtn?.contains(event.target) && !els.reasoningMenu.contains(event.target)) {
      closeReasoningMenu();
    }
  }
  if (!els.questionMenu || els.questionMenu.hidden) return;
  if (event.target === els.questionMenuBtn || els.questionMenuBtn.contains(event.target)) return;
  if (els.questionMenu.contains(event.target)) return;
  closeQuestionMenu();
});
document.addEventListener("mousemove", (event) => {
  if (!els.questionMenu || els.questionMenu.hidden) return;
  if (isInsideQuestionMenuArea(event.target)) {
    cancelQuestionMenuClose();
  } else {
    scheduleQuestionMenuClose();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.querySelector("#serviceShellModal")?.hidden) {
    closeServiceShell();
  }
  if (event.key === "Escape") closeMobileDrawers();
});
window.addEventListener("resize", () => {
  if (window.innerWidth > 1180) closeMobileDrawers();
  if (isPhoneLayout()) {
    const current = els.logbar?.getBoundingClientRect().width || mobileLogbarBounds().preferred;
    setMobileLogbarWidth(current, { persist: false });
  }
});

restoreMobileLogbarWidth();
init().catch((error) => {
  // If the JSON data cannot be loaded, keep the page visible and give a clear
  // local-service hint instead of leaving an empty shell.
  console.error(error);
  els.messages.innerHTML = '<div class="welcome"><div class="welcome-title">数据加载失败</div><p>请确认本地服务已启动。</p></div>';
});
