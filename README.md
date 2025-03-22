# Automated File Validation System  

## Overview  
This project automates file validation using Python, FastAPI, and SQL. It detects data anomalies, incorrect delimiters, and invalid formats.

## Features  
- Automated delimiter detection using NLP  
- Regex-based validation  
- Interactive feedback system  
- Web UI for validation and feedback submission  

## Installation  
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-username/your-repository.git
   
2. Navigate into the project directory:
   ```sh
   cd your-repository
3. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
5. Run the project:
   ```sh
   python main.py

## Usage:
- Upload files for validation through the web interface.
- View detected anomalies and provide feedback.
- The system refines regex patterns dynamically based on feedback.

## Technologies Used:
- Python – Core programming language
- FastAPI – Backend framework
- SQL – Database for storing feedback
- NLP – Used for column type detection
- Regex – For structured validation
- HTML, CSS, JavaScript – Web interface

## Contributing: 
If you'd like to contribute, feel free to fork the repository, create a feature branch, and submit a pull request.

## License:
This project is open-source under the MIT License.
