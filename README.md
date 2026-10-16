# 📚 Bookworm — Book Recommendation System

**Discover your next favourite book with data mining!**

Bookworm is a book recommendation system that uses **Apriori Association Rule Mining** to discover relationships between readers' book preferences and recommend books based on the books they already enjoy.

Built with Python, Flask, Pandas, NumPy, HTML, CSS, and JavaScript, Bookworm combines data mining with an interactive, colourful web interface to make book discovery engaging and intuitive.

## ✨ Features

- 📖 **Book Discovery** — Search and explore a curated catalog of 500 popular books.
- 💜 **Personalized Recommendations** — Select books you enjoy to discover related books.
- 🧠 **Apriori Algorithm** — Identifies frequent itemsets and association rules from reader-rating transactions.
- 📊 **Recommendation Insights** — Understand recommendations through support, confidence, and lift.
- 🔍 **Interactive Search** — Find books quickly using their titles.
- 🎨 **Colourful Web Interface** — Enjoy a modern, responsive interface built with HTML, CSS, and JavaScript.
- ⚡ **Efficient Rule Lookup** — Uses precomputed association rules to generate recommendations without rerunning the mining algorithm for every request.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming and data processing |
| Pandas | Dataset preparation and manipulation |
| NumPy | Efficient numerical operations and Apriori calculations |
| Flask | Backend and web application |
| HTML5 | Web page structure |
| CSS3 | Styling and responsive design |
| JavaScript | Interactive frontend |
| Git & GitHub | Version control and collaboration |

## 🧠 How It Works

Bookworm applies Apriori Association Rule Mining to book-rating data.

1. **Data Preparation:** Clean and prepare the book-rating dataset.
2. **Transaction Creation:** Treat each reader's positively rated books as a transaction, using a rating of 4 or 5 as a positive preference.
3. **Frequent Itemset Mining:** Identify groups of books frequently enjoyed by the same readers.
4. **Association Rule Generation:** Generate rules that describe relationships between books.
5. **Recommendation:** Match a user's selected books against the learned rules and rank suitable recommendations.

### 📐 Measures Used

- **Support:** How frequently books in a rule occur together among all reader transactions.
- **Confidence:** How often the recommended book appears among readers who liked the antecedent book or books.
- **Lift:** How strongly the books are associated compared with the recommended book's overall popularity.

### Model Configuration

| Parameter | Value |
|---|---:|
| Minimum support | 1% |
| Minimum confidence | 30% |
| Maximum itemset length | 3 |
| Lift filter | Greater than 1.0 |
| Selected book catalog | 500 books |

The recommendation engine uses precomputed association rules, allowing the website to retrieve recommendations without mining the complete dataset for every user request.

## 📊 Dataset

Bookworm uses the **goodbooks-10k** dataset, which contains book metadata and reader ratings.

- **Books:** 10,000
- **Ratings:** Approximately 6 million
- **Selected catalog:** 500 books
- **Positive preference:** Rating ≥ 4

Dataset source: [zygmuntz/goodbooks-10k on GitHub](https://github.com/zygmuntz/goodbooks-10k)

The dataset repository identifies its data under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license. Please consult the original repository and license terms for attribution and redistribution requirements.

## 🚀 Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.12 or another compatible Python version
- Git
- A modern web browser

### 1. Clone the Repository

```bash
git clone https://github.com/alnamariapamcracious/Bookworm.git
cd Bookworm
```

### 2. Create a Virtual Environment (Recommended)

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

Install the required packages:

```bash
python -m pip install flask pandas numpy
```

If a `requirements.txt` file is available in the repository, you can instead run:

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the `books.csv` and `ratings.csv` files from the [goodbooks-10k repository](https://github.com/zygmuntz/goodbooks-10k).

Place the files in the following directory structure:

```text
data/
└── raw/
    ├── books.csv
    └── ratings.csv
```

Create the `data/raw/` directory if it does not already exist.

Then run:

```bash
python prepare_dataset.py
```

This prepares the processed data required by the recommendation system.

### 5. Generate Association Rules

Run the Apriori mining script:

```bash
python mine_apriori.py
```

This generates the frequent itemsets and association rules used by the recommendation engine.

**Note:** Mining the full prepared dataset can take several minutes. You only need to run this step when generating or regenerating the model outputs.

### 6. Launch the Website

Start the Flask application:

```bash
python app.py
```

Open your browser and visit:

**http://127.0.0.1:5000**

You can now search for books, select your favourites, and explore the resulting recommendations.

> **Important:** The application needs its expected processed data and association-rule files. A fresh clone will not work until these files are generated or otherwise supplied. Large datasets and generated CSV files may be excluded from Git to keep the repository lightweight.

## 📁 Project Structure

```text
Bookworm/
├── app.py                  # Flask web application
├── prepare_dataset.py      # Dataset preparation
├── mine_apriori.py         # Apriori algorithm
├── requirements.txt        # Python dependencies, if included
├── README.md               # Project documentation
├── .gitignore              # Files excluded from Git
├── data/
│   ├── raw/                # Original dataset files
│   └── processed/          # Prepared data and model outputs
├── templates/
│   └── index.html          # Main web page
└── static/
    ├── css/
    │   └── style.css       # Website styling
    ├── js/
    │   └── app.js          # Frontend interactions
    └── img/
        └── book_placeholder.svg
```

## 🎯 Project Objectives

- Apply association rule mining to a real-world book-rating dataset.
- Discover meaningful relationships among reader preferences.
- Build a practical recommendation system using frequent itemsets.
- Demonstrate the use of support, confidence, and lift.
- Integrate a data-mining algorithm with an interactive web application.

## 🔮 Future Improvements

- Genre-based filtering and sorting.
- More advanced recommendation ranking.
- User accounts and saved reading lists.
- Improved evaluation of recommendation quality.
- Expanded book catalogs and additional recommendation methods.

## 🎓 About

Bookworm was developed as a **college Data Mining project** to demonstrate the practical application of the Apriori algorithm in a book recommendation system.

## 📜 Acknowledgements

- The contributors to the [goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k).
- The Python, Flask, Pandas, and NumPy communities.
- Everyone who contributed to the development and testing of Bookworm.

---

**Made with ❤️ for book lovers and data mining enthusiasts.**
