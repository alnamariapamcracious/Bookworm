/**
 * app.js
 * ------
 * Client-Side Application Controller for:
 * "Book Taste & Recommendation Mining"
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    selectedBooks: new Map(), // book_id -> book object
    catalog: [],
    searchDebounceTimer: null,
    isLoading: false
  };

  // DOM Elements
  const catalogList = document.getElementById("catalogList");
  const catalogSearchInput = document.getElementById("catalogSearchInput");
  const searchClearBtn = document.getElementById("searchClearBtn");
  const catalogCount = document.getElementById("catalogCount");

  const bookshelfTray = document.getElementById("bookshelfTray");
  const emptyTrayMsg = document.getElementById("emptyTrayMsg");
  const shelfCount = document.getElementById("shelfCount");
  const clearShelfBtn = document.getElementById("clearShelfBtn");

  const recsContent = document.getElementById("recsContent");
  const recsCount = document.getElementById("recsCount");

  // Preset Buttons
  const presetHarryPotter = document.getElementById("presetHarryPotter");
  const presetHungerGames = document.getElementById("presetHungerGames");
  const presetVampire = document.getElementById("presetVampire");
  const presetClassics = document.getElementById("presetClassics");

  // =========================================================================
  // 1. Initial Load & Catalog Fetch
  // =========================================================================
  async function loadInitialData() {
    await fetchStats();
    await fetchBooks();
  }

  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      if (data.success && data.stats) {
        const s = data.stats;
        const statUsers = document.getElementById("statUsers");
        const statBooks = document.getElementById("statBooks");
        const statRules = document.getElementById("statRules");

        if (statUsers) statUsers.textContent = s.transactions_analyzed.toLocaleString();
        if (statBooks) statBooks.textContent = s.catalog_books.toLocaleString();
        if (statRules) statRules.textContent = s.total_rules.toLocaleString();
      }
    } catch (err) {
      console.warn("Could not load model stats:", err);
    }
  }

  async function fetchBooks(query = "") {
    try {
      const url = query ? `/api/books?q=${encodeURIComponent(query)}&limit=100` : "/api/books?limit=100";
      const res = await fetch(url);
      const data = await res.json();

      if (data.success) {
        state.catalog = data.books;
        catalogCount.textContent = `${data.total_matched} books`;
        renderCatalog();
      }
    } catch (err) {
      console.error("Error fetching catalog books:", err);
      catalogList.innerHTML = `<div class="empty-state"><p>Error loading book catalog. Please refresh.</p></div>`;
    }
  }

  // =========================================================================
  // 2. Render Book Catalog
  // =========================================================================
  function renderCatalog() {
    catalogList.innerHTML = "";

    if (state.catalog.length === 0) {
      catalogList.innerHTML = `
        <div class="empty-state" style="padding: 2.5rem 1rem;">
          <p>No books found matching your search. Try another author or title!</p>
        </div>
      `;
      return;
    }

    const fragment = document.createDocumentFragment();

    state.catalog.forEach((book) => {
      const isSelected = state.selectedBooks.has(book.book_id);

      const item = document.createElement("div");
      item.className = `catalog-item ${isSelected ? "selected" : ""}`;
      item.id = `catalog-item-${book.book_id}`;

      // Safe cover URL with placeholder fallback
      const coverUrl = book.small_image_url || book.image_url || "/img/book_placeholder.svg";

      item.innerHTML = `
        <img 
          src="${escapeHtml(coverUrl)}" 
          alt="${escapeHtml(book.title)}" 
          class="catalog-thumb" 
          loading="lazy"
          onerror="this.onerror=null; this.src='/img/book_placeholder.svg';"
        >
        <div class="catalog-info">
          <div class="catalog-title" title="${escapeHtml(book.title)}">${escapeHtml(book.title)}</div>
          <div class="catalog-author">${escapeHtml(book.authors)}</div>
          <div class="catalog-meta">
            <span class="catalog-rating">★ ${book.average_rating ? book.average_rating.toFixed(2) : "N/A"}</span>
            <span>•</span>
            <span>${book.original_publication_year || "N/A"}</span>
            <span>•</span>
            <span>${book.positive_user_count ? book.positive_user_count.toLocaleString() : "0"} likes</span>
          </div>
        </div>
        <button 
          type="button" 
          class="catalog-add-btn" 
          data-book-id="${book.book_id}"
          title="${isSelected ? 'Remove from Bookshelf' : 'Add to Bookshelf'}"
        >
          ${isSelected ? "✓ Added" : "+ Add"}
        </button>
      `;

      const btn = item.querySelector(".catalog-add-btn");
      btn.addEventListener("click", () => {
        toggleBookSelection(book);
      });

      fragment.appendChild(item);
    });

    catalogList.appendChild(fragment);
  }

  // =========================================================================
  // 3. Bookshelf Management (Add / Remove)
  // =========================================================================
  function toggleBookSelection(book) {
    if (state.selectedBooks.has(book.book_id)) {
      state.selectedBooks.delete(book.book_id);
    } else {
      state.selectedBooks.set(book.book_id, book);
    }
    updateBookshelfUI();
    fetchRecommendations();
  }

  function removeBook(bookId) {
    state.selectedBooks.delete(bookId);
    updateBookshelfUI();
    fetchRecommendations();
  }

  function clearBookshelf() {
    state.selectedBooks.clear();
    updateBookshelfUI();
    fetchRecommendations();
  }

  function updateBookshelfUI() {
    const count = state.selectedBooks.size;
    shelfCount.textContent = `${count} selected`;

    // Clear and re-render chips
    bookshelfTray.innerHTML = "";

    if (count === 0) {
      bookshelfTray.appendChild(emptyTrayMsg);
      clearShelfBtn.style.display = "none";
    } else {
      clearShelfBtn.style.display = "inline-flex";

      state.selectedBooks.forEach((book) => {
        const chip = document.createElement("div");
        chip.className = "book-chip";
        chip.innerHTML = `
          <span>📖 ${escapeHtml(book.title)}</span>
          <button type="button" class="chip-remove" title="Remove ${escapeHtml(book.title)}">&times;</button>
        `;
        chip.querySelector(".chip-remove").addEventListener("click", () => {
          removeBook(book.book_id);
        });
        bookshelfTray.appendChild(chip);
      });
    }

    // Update catalog buttons highlight state
    document.querySelectorAll(".catalog-item").forEach((el) => {
      const bId = parseInt(el.id.replace("catalog-item-", ""), 10);
      const btn = el.querySelector(".catalog-add-btn");
      if (state.selectedBooks.has(bId)) {
        el.classList.add("selected");
        if (btn) btn.textContent = "✓ Added";
      } else {
        el.classList.remove("selected");
        if (btn) btn.textContent = "+ Add";
      }
    });
  }

  // =========================================================================
  // 4. Recommendation Engine API Call
  // =========================================================================
  async function fetchRecommendations() {
    const selectedIds = Array.from(state.selectedBooks.keys());

    if (selectedIds.length === 0) {
      recsCount.textContent = "0 found";
      recsContent.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-art">
            <div class="floating-bubble bubble-1">📚</div>
            <div class="floating-bubble bubble-2">✨</div>
            <div class="floating-bubble bubble-3">💡</div>
          </div>
          <h3>Your Personalized Bookshelf is Empty</h3>
          <p>Select at least one book from the catalog on the left to trigger the Apriori recommendation engine.</p>
        </div>
      `;
      return;
    }

    // Show colorful loading skeleton
    recsContent.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-art">
          <div class="floating-bubble bubble-1" style="animation: spin 1.2s infinite linear;">⚙️</div>
          <div class="floating-bubble bubble-2">✨</div>
          <div class="floating-bubble bubble-3">📚</div>
        </div>
        <h3>Mining Apriori Rules...</h3>
        <p>Scanning 83,314 association rules for matches with your selected taste basket.</p>
      </div>
    `;

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ book_ids: selectedIds })
      });

      const data = await res.json();
      renderRecommendations(data);
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      recsContent.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚠️</div>
          <h3>Recommendation Request Failed</h3>
          <p>An error occurred while communicating with the mining engine. Please try again.</p>
        </div>
      `;
    }
  }

  // =========================================================================
  // 5. Render Recommendation Cards
  // =========================================================================
  function renderRecommendations(data) {
    if (!data.success || !data.recommendations || data.recommendations.length === 0) {
      recsCount.textContent = "0 found";
      recsContent.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-art">
            <div class="floating-bubble bubble-1">🔍</div>
            <div class="floating-bubble bubble-2">✨</div>
            <div class="floating-bubble bubble-3">📖</div>
          </div>
          <h3>No Exact Apriori Rules Matched</h3>
          <p>${escapeHtml(data.message || "No frequent co-occurrence patterns met the 1% support and 30% confidence thresholds for this specific combination.")}</p>
        </div>
      `;
      return;
    }

    recsCount.textContent = `${data.recommendation_count} recommended`;

    const grid = document.createElement("div");
    grid.className = "recs-grid";

    data.recommendations.forEach((rec) => {
      const card = document.createElement("div");
      card.className = "rec-card";

      const coverUrl = rec.image_url || rec.small_image_url || "/img/book_placeholder.svg";
      const confidencePct = (rec.confidence * 100).toFixed(1);
      const liftVal = rec.lift.toFixed(1);
      const supportPct = (rec.support * 100).toFixed(2);

      let supportingText = "";
      if (rec.total_supporting_rules > 1) {
        supportingText = `<span class="supporting-rules-badge">⚡ Supported by ${rec.total_supporting_rules} distinct association rules</span>`;
      }

      const formattedAntecedent = escapeHtml(rec.matched_antecedent_titles.replace(/ \| /g, " + "));

      card.innerHTML = `
        <div class="rec-card-top">
          <img 
            src="${escapeHtml(coverUrl)}" 
            alt="${escapeHtml(rec.title)}" 
            class="rec-cover" 
            loading="lazy"
            onerror="this.onerror=null; this.src='/img/book_placeholder.svg';"
          >
          <div class="rec-info">
            <h3 class="rec-title" title="${escapeHtml(rec.title)}">${escapeHtml(rec.title)}</h3>
            <div class="rec-author">${escapeHtml(rec.authors)}</div>
            <div class="rec-rating">
              ★ ${rec.average_rating ? rec.average_rating.toFixed(2) : "N/A"}
              <span style="color: var(--text-dim); margin-left: 0.35rem;">(${rec.original_publication_year || "Unknown"})</span>
            </div>
          </div>
        </div>

        <div class="antecedent-pill-tag">
          <span class="trigger-label">Triggered By:</span>
          <span class="trigger-books">${formattedAntecedent}</span>
        </div>

        <div class="metrics-row">
          <span class="metric-pill lift" title="Lift measures correlation strength above random co-occurrence">
            🔥 ${liftVal}x Lift
          </span>
          <span class="metric-pill confidence" title="Confidence: Probability of liking this book given your antecedent">
            🎯 ${confidencePct}% Conf
          </span>
          <span class="metric-pill support" title="Support: Total percentage of readers who loved this combination">
            📊 ${supportPct}% Supp
          </span>
        </div>

        <div class="rule-explanation">
          <strong>Why Recommended?</strong>
          ${escapeHtml(rec.explanation)}
          ${supportingText}
        </div>
      `;

      grid.appendChild(card);
    });

    recsContent.innerHTML = "";
    recsContent.appendChild(grid);
  }

  // =========================================================================
  // 6. Search Event Handlers (Debounced)
  // =========================================================================
  catalogSearchInput.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    searchClearBtn.style.display = val ? "flex" : "none";

    clearTimeout(state.searchDebounceTimer);
    state.searchDebounceTimer = setTimeout(() => {
      fetchBooks(val);
    }, 250);
  });

  searchClearBtn.addEventListener("click", () => {
    catalogSearchInput.value = "";
    searchClearBtn.style.display = "none";
    fetchBooks("");
  });

  clearShelfBtn.addEventListener("click", clearBookshelf);

  // =========================================================================
  // 7. Reading Taste Presets (One-Click Demonstrations)
  // =========================================================================
  function applyPreset(bookIds) {
    state.selectedBooks.clear();
    bookIds.forEach((id) => {
      const book = state.catalog.find((b) => b.book_id === id);
      if (book) {
        state.selectedBooks.set(id, book);
      } else {
        // Query server for specific book if not in current view
        fetch(`/api/books?limit=500`)
          .then((r) => r.json())
          .then((data) => {
            if (data.success) {
              const found = data.books.find((b) => b.book_id === id);
              if (found) state.selectedBooks.set(id, found);
              updateBookshelfUI();
              fetchRecommendations();
            }
          });
      }
    });
    updateBookshelfUI();
    fetchRecommendations();
  }

  // Preset 1: Harry Potter (Books 2 & 18 -> Sorcerer's Stone & Prisoner of Azkaban)
  presetHarryPotter.addEventListener("click", () => applyPreset([2, 18]));

  // Preset 2: Dystopian Hits (Books 1 & 11 -> Hunger Games & Kite Runner)
  presetHungerGames.addEventListener("click", () => applyPreset([1, 11]));

  // Preset 3: Urban Fantasy (Books 197 & 563 -> Dead Until Dark & Club Dead)
  presetVampire.addEventListener("click", () => applyPreset([197, 563]));

  // Preset 4: Classics (Books 4 & 10 -> To Kill a Mockingbird & Pride and Prejudice)
  presetClassics.addEventListener("click", () => applyPreset([4, 10]));

  // =========================================================================
  // Utility Functions
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Spin animation for loader
  const styleSheet = document.createElement("style");
  styleSheet.textContent = `@keyframes spin { 100% { transform: rotate(360deg); } }`;
  document.head.appendChild(styleSheet);

  // Initialize
  loadInitialData();
});
