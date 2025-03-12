# CureNow Chatbot - README

## Overview

The CureNow Chatbot is a Flask-based application designed to provide medical information and assistance to users. It integrates several key features, including:

-   **Chat Interface**: A user-friendly interface for interacting with the chatbot.
-   **Retrieval-Augmented Generation (RAG)**: Uses Pinecone for vector storage and Google Gemini for generating responses, enhancing the quality and relevance of the chatbot's answers.
-   **Spell Checker**: Corrects user input to ensure accurate and relevant responses.
-   **Image-Based Disease Prediction**: Integrates a PyTorch-based multi-head model for predicting diseases from medical images (brain, lung, and skin).
-   **Chat History**: Stores and retrieves chat history using a PostgreSQL database.
-   **Chat Export**: Exports chat history in various formats (TXT, DOCX, PDF).

## Technologies Used

-   **Flask**: Web framework for building the application.
-   **Langchain**: Framework for building language model applications.
-   **Pinecone**: Vector database for storing and retrieving document embeddings.
-   **Google Gemini**: Large language model for generating responses.
-   **Hugging Face Transformers**: For embeddings and sequence classification models.
-   **PyTorch**: Deep learning framework for the image-based disease prediction model.
-   **PostgreSQL**: Database for storing chat history and appointment information.
-   **SpellChecker**: For auto-correcting user input.
-   **python-dotenv**: For loading environment variables from a `.env` file.
-   **docx, fpdf**: For exporting chat history to DOCX and PDF formats.

## Setup and Installation

### Prerequisites

-   Python 3.7+
-   PostgreSQL database
-   API keys for Pinecone and Google Gemini

### Installation Steps

1.  **Clone the repository:**

    ```
    git clone https://github.com/ChandekarDhruvin/curenow.git

    ```

2.  **Create a virtual environment:**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```
    pip install -r requirements.txt
    ```

4.  **Set up the environment variables:**

    -   Create a `.env` file in the root directory.
    -   Add the following variables:

        ```
        SECRET_KEY=<your_secret_key>
        PINECONE_API_KEY=<your_pinecone_api_key>
        GEMINI_API_KEY=<your_gemini_api_key>
        DATABASE_URL=<your_database_url> # Example: postgresql://user:password@host:port/database
        ```

5.  **Download Hugging Face embeddings (if not already present):**

    -   The application uses Hugging Face embeddings. Ensure the necessary model is downloaded:

        ```
        from static.src.helper import download_hugging_face_embeddings
        download_hugging_face_embeddings()
        ```

6.  **Run the Flask application:**

    ```
    python app.py
    ```

    The application will start running on `http://127.0.0.1:8000/` or a similar address.

## Database Setup

1.  **Install PostgreSQL:**

    -   Follow the instructions on the [official PostgreSQL website](https://www.postgresql.org/) to install PostgreSQL on your system.

2.  **Create a database:**

    -   Connect to the PostgreSQL server using a tool like `psql` or pgAdmin.
    -   Create a new database for the application:

        ```
        CREATE DATABASE <database_name>;
        ```

3.  **Update the `.env` file:**

    -   Set the `DATABASE_URL` environment variable to the connection string for your PostgreSQL database.
        Example:

        ```
        DATABASE_URL=postgresql://username:password@localhost:5432/databasename
        ```

## Model Integration

### Image-Based Disease Prediction Model

-   The application integrates a PyTorch-based multi-head model for predicting diseases from medical images.
-   The model supports the following tasks:
    -   Brain tumor detection (Glioma, Meningioma, No Tumor, Pituitary)
    -   Lung disease detection (Corona Virus Disease, Normal, Pneumonia, Tuberculosis)
    -   Skin disease detection (Actinic keratosis, Dermatofibroma, Squamous cell carcinoma, Atopic Dermatitis, Melanocytic nevus, Tinea Ringworm Candidiasis, Benign keratosis, Melanoma, Vascular lesion)
-   The model weights are loaded from the `x-rayy/Multi_head_medical_model.pth` file.
-   Ensure that the file exists at the specified path.

### FastAPI Integration

-   The image processing functionality (originally in FastAPI) is now integrated directly into the Flask app.
-   The `predict_image` function handles image processing and prediction.

## Usage

1.  **Access the Chatbot:**

    -   Open a web browser and navigate to the address where the Flask application is running (e.g., `http://127.0.0.1:5000/`).

2.  **Chat Interface:**

    -   Type your message in the input box and press Enter or click the Send button.
    -   The chatbot will respond with relevant information.

3.  **Image-Based Disease Prediction:**

    -   Upload a medical image (brain, lung, or skin X-ray) using the file upload feature.
    -   Select the appropriate task (brain, lung, or skin).
    -   The chatbot will predict the disease based on the image.


4.  **Chat History:**

    -   The chat history is stored in the database and displayed on the main page.

5.  **Export Chat:**

    -   Click the "Export Chat" button to export the chat history in TXT, DOCX, or PDF format.

## API Endpoints

-   `/`: Renders the main page with the chat interface.
-   `/get` (POST): Handles text-based chat messages.
-   `/get_response` (POST): Handles both text and image-based requests.
-   `/clear_chats` (POST): Clears all chat history from the database.
-   `/export_chat` (GET): Exports the chat history in various formats.

## Notes

-   Ensure that all API keys and database URLs are correctly set in the `.env` file.
-   The image-based disease prediction model requires significant computational resources. Consider using a GPU for faster processing.
-   The application uses cookies to maintain session state. Ensure that cookies are enabled in your browser.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License

[MIT License](LICENSE)


![image](https://github.com/user-attachments/assets/db724a3d-0865-4f20-8e3d-b3ec62d76d9f)

