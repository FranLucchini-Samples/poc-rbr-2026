// Paddle-style coverflow gallery. Fetches the image list, lays the cards out in
// a 3D fan, and lets the user flick through and select one.

const track = document.getElementById("track");
const emptyMsg = document.getElementById("empty");
const prevBtn = document.getElementById("prev");
const nextBtn = document.getElementById("next");
const selectBtn = document.getElementById("select");
const selectionLabel = document.getElementById("selection");
const caption = document.getElementById("caption");

let images = [];
let cards = [];
let current = 0;

// Geometry of the fanned-out "paddle".
const SPREAD = 280; // horizontal px between neighbouring cards
const DEPTH = 120; // how far back side cards sit (px)
const ROTATION = 42; // yaw of side cards (deg)
const VISIBLE = 4; // cards shown on each side of the centre

async function init() {
  const res = await fetch("/api/images");
  const data = await res.json();
  images = data.images;

  if (images.length === 0) {
    emptyMsg.hidden = false;
    prevBtn.disabled = nextBtn.disabled = true;
    return;
  }

  buildCards();
  await syncSelectionFromServer();
  layout();
  bindEvents();
}

function buildCards() {
  track.innerHTML = "";
  cards = images.map((img, i) => {
    const card = document.createElement("div");
    card.className = "card";

    const el = document.createElement("img");
    el.loading = "lazy";
    el.src = img.thumb;
    el.alt = img.name;
    el.addEventListener("error", () => card.classList.add("broken"), {
      once: true,
    });

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = img.name;

    card.append(el, label);
    card.addEventListener("click", () => {
      if (i === current) {
        selectCurrent();
      } else {
        goTo(i);
      }
    });
    track.appendChild(card);
    return card;
  });
}

function layout() {
  cards.forEach((card, i) => {
    const offset = i - current;
    const abs = Math.abs(offset);

    if (abs > VISIBLE) {
      card.style.opacity = "0";
      card.style.pointerEvents = "none";
      card.style.transform = `translateX(${Math.sign(offset) * SPREAD * (VISIBLE + 1)}px) translateZ(-600px)`;
      return;
    }

    const translateX = offset * SPREAD;
    const translateZ = -abs * DEPTH;
    const rotateY = offset === 0 ? 0 : -Math.sign(offset) * ROTATION;
    const scale = offset === 0 ? 1.08 : 1;

    card.style.opacity = String(Math.max(0.35, 1 - abs * 0.18));
    card.style.pointerEvents = "auto";
    card.style.zIndex = String(100 - abs);
    card.style.transform =
      `translateX(${translateX}px) translateZ(${translateZ}px) ` +
      `rotateY(${rotateY}deg) scale(${scale})`;
    card.classList.toggle("active", offset === 0);
  });

  caption.textContent = `${current + 1} / ${images.length} · ${images[current].name}`;
}

function goTo(index) {
  current = Math.max(0, Math.min(images.length - 1, index));
  layout();
}

function bindEvents() {
  prevBtn.addEventListener("click", () => goTo(current - 1));
  nextBtn.addEventListener("click", () => goTo(current + 1));
  selectBtn.addEventListener("click", selectCurrent);
  selectBtn.disabled = false;

  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") goTo(current - 1);
    else if (e.key === "ArrowRight") goTo(current + 1);
    else if (e.key === "Enter") selectCurrent();
  });

  // Trackpad / mouse-wheel flicking.
  let wheelLock = false;
  track.parentElement.addEventListener(
    "wheel",
    (e) => {
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      if (Math.abs(delta) < 8 || wheelLock) return;
      e.preventDefault();
      wheelLock = true;
      goTo(current + (delta > 0 ? 1 : -1));
      setTimeout(() => (wheelLock = false), 220);
    },
    { passive: false }
  );

  // Basic touch swipe.
  let startX = null;
  const stage = track.parentElement;
  stage.addEventListener("touchstart", (e) => (startX = e.touches[0].clientX), {
    passive: true,
  });
  stage.addEventListener(
    "touchend",
    (e) => {
      if (startX === null) return;
      const dx = e.changedTouches[0].clientX - startX;
      if (Math.abs(dx) > 40) goTo(current + (dx < 0 ? 1 : -1));
      startX = null;
    },
    { passive: true }
  );
}

async function selectCurrent() {
  const name = images[current].name;
  const res = await fetch("/api/selection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (res.ok) window.location.href = "/detail";
}

function markSelected(name) {
  selectionLabel.textContent = `Selected: ${name}`;
  selectionLabel.classList.add("active");
}

async function syncSelectionFromServer() {
  try {
    const res = await fetch("/api/selection");
    const data = await res.json();
    if (data.name) {
      markSelected(data.name);
      const idx = images.findIndex((img) => img.name === data.name);
      if (idx >= 0) current = idx;
    }
  } catch (_) {
    /* first load, no selection yet */
  }
}

init();
