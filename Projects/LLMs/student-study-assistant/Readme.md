# 🎓 Student Study Assistant with RAG

A smart document-based Q&A system that uses **Retrieval-Augmented Generation (RAG)** to answer questions from text documents. Built as part of CodeChef's Weekend Challenge.

## 📋 Overview

This project implements a two-part AI-powered study assistant:
- **Part 1**: Document loading, text chunking, and keyword-based retrieval
- **Part 2**: Integration with Groq's LLM API for natural language answer generation

The system finds the most relevant passage from a document and generates context-aware answers using the Llama 3.3 model.

## ✨ Features

- 📄 **Document Processing**: Load and parse text files
- 🔪 **Smart Chunking**: Split documents into overlapping chunks for better context
- 🎯 **Keyword Matching**: Score and retrieve the most relevant text passage
- 🤖 **LLM Integration**: Generate natural language answers using Groq API
- 💡 **Context-Aware**: Answers are grounded in the provided document

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-study-assistant.git
cd student-study-assistant
```

2. Install required dependencies:
```bash
pip install groq
```

### Setup Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create API Key** and copy it
5. Keep it safe - you'll paste it when running the script

## 📖 Usage

1. Create a `document.txt` file with your study material

2. Run the program:
```bash
python main.py
```

3. Enter your question when prompted:
```
Enter your question: What is water cycle?
```

4. Provide your Groq API key:
```
Enter your Groq API key: ********************************************
```

5. Get your answer:
```
Answer: The water cycle is the process by which water moves through 
the planet, supporting life, and has several main stages: evaporation, 
condensation, precipitation, and collection...
```

## 🏗️ Project Structure

```
student-study-assistant/
│
├── main.py              # Main application (Part 1 & 2)
├── document.txt         # Your study document
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## 🔧 How It Works

### Part 1: Retrieval System

1. **Load Document**: Reads text from `document.txt`
2. **Chunk Text**: Splits document into 300-word chunks with 50-word overlap
3. **Score Chunks**: Matches question keywords with chunk content
4. **Retrieve Best**: Selects the chunk with highest keyword overlap

### Part 2: Answer Generation

5. **Build Prompt**: Formats context and question for the LLM
6. **Call Groq API**: Sends prompt to Llama 3.3 70B model
7. **Generate Answer**: Returns natural language response

## 🛠️ Configuration

You can customize the following parameters in the code:

- `chunk_size`: Maximum words per chunk (default: 300)
- `overlap`: Overlapping words between chunks (default: 50)
- `temperature`: LLM creativity level (default: 0.3)
- `max_completion_tokens`: Maximum response length (default: 512)

## 📝 Example

**Document**: Information about the water cycle

**Question**: "What is water cycle?"

**Retrieved Context**: Most relevant paragraph about water cycle stages

**Generated Answer**: Natural language explanation combining context and question

## 🎯 CodeChef Weekend Challenge

This project was built as part of CodeChef's Weekend Challenge to learn:
- Document processing and text chunking
- Keyword-based information retrieval
- RAG (Retrieval-Augmented Generation) systems
- LLM API integration

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CodeChef](https://www.codechef.com/) for the weekend challenge
- [Groq](https://groq.com/) for providing the LLM API
- [Meta AI](https://ai.meta.com/) for the Llama 3.3 model

## 📧 Contact

Your Name - Madhav Deshatwad

Project Link: [https://github.com/sdmadhav/student-study-assistant](https://github.com/sdmadhav/student-study-assistant)

---

⭐ If you found this helpful, please consider giving it a star!
