import os
import re
import csv
import json
import spacy
import torch
import difflib  # For fuzzy matching
import chardet
import numpy as np
import pandas as pd
from collections import Counter
from torch import nn
from typing import List
from spacy.lang.en import English
from openpyxl import load_workbook
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, UploadFile, File, Form, Request
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, TensorDataset
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Initialize FastAPI app
app = FastAPI()

# Path to store uploaded files and feedback
UPLOAD_DIR = r"C:\ZingMind_December2024\Regexify\uploads"

# MySQL Database Connection (Update with your credentials)
DATABASE_URL = "mysql+pymysql://root:NpGkPg#6@localhost/File_validation_feedback?charset=utf8mb4"

# Create SQLAlchemy Engine and Base
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Load models
# Load pre-trained model
nlm_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = English()

# Declare global variable
LATEST_FILE_PATH = None

# Define Feedback Table (matching the table in your database)
class Feedback(Base):
    __tablename__ = "Feedback_table"  # Ensure this matches the table name in MySQL
    User_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    Column_name = Column(String(255), nullable=True)
    Change_type = Column(Text, nullable=True)
    feedback_text = Column(Text, nullable=True)

# Create Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#Function to detect delimiter
def symbol_delimiter(lines):

    # Combine lines for better statistical analysis
    combined_text = "\n".join(lines)

    # Extract potential delimiters: Include spaces and tabs separately
    potential_delimiters = re.findall(r"[^\w]", combined_text)  # Includes special chars, spaces, and tabs

    # Count occurrences of each delimiter
    delimiter_counts = Counter(potential_delimiters)

    # Handle cases where tab (`\t`) or space (` `) might be the delimiter
    if delimiter_counts.get("\t", 0) > 0:
        return "\t"  # Prioritize tab if detected

    if delimiter_counts.get(" ", 0) > 0:
        # Count spaces in each line
        space_count_per_line = [line.count(" ") for line in lines if line.strip()]

        # Find the most common space count
        common_space_count = Counter(space_count_per_line).most_common(1)

        if common_space_count:
            most_common_spaces, freq = common_space_count[0]

            # Ensure at least 70% of the lines have the same space count
            if freq / len(space_count_per_line) >= 0.7 and most_common_spaces > 1:
                # Ensure spaces are not just separating words
                non_alpha_lines = sum(1 for line in lines if not re.match(r'^[a-zA-Z\s]+$', line.strip()))

                if non_alpha_lines / len(lines) > 0.5:  # At least 50% of lines should not be pure words
                    return " "

    if not delimiter_counts:
        return None  # No delimiter found

    # Find the most common single-character delimiter
    most_common_single = max(delimiter_counts, key=delimiter_counts.get)

    return most_common_single  # Return the most frequent delimiter

def extract_values(line, delimiter): 
    nlp = spacy.blank("en")  # Initialize spaCy NLP model
    
    # Ensure the line contains the delimiter
    if delimiter not in line:
        raise ValueError("Delimiter not found in the line")
    
    # Using regex to correctly extract values inside double quotes
    pattern = r'"(.*?)"'
    matches = re.findall(pattern, line)
    
    # Process extracted values with NLP
    processed_values = []
    for value in matches:
        if value.strip() == delimiter:  # Skip delimiter-like values
            continue  
        doc = nlp(value)
        processed_values.append(" ".join([token.text for token in doc]))  # Tokenized extraction
    
    return processed_values 

def Corrected_file(file_path, detected_delimiter, flag):
    try:
        Validation_file = "Corrected_File.csv"
        with open(file_path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()
            if not lines:
                return {"error": "Empty file or unreadable content"}
            data = []
            if flag == 1:            
                for line in lines:
                    try:
                        extracted_values = extract_values(line, detected_delimiter)
                        if extracted_values:
                            data.append(extracted_values)  # Append extracted values correctly
                    except ValueError as e:
                        return {"error": str(e)}
            else:
                # Ensure correct splitting of tab-separated values
                headers = [header.strip('"') for header in lines[0].strip().split(detected_delimiter)]
                data_rows = [[value.strip('"') for value in line.strip().split(detected_delimiter)] for line in lines[1:]]

                if not data_rows:
                    return {"error": "No valid data rows found"}

                data.append(headers)  # Ensure headers are included
                data.extend(data_rows)  # Append data rows
                
               

            if not data:
                return {"error": "Empty or malformed CSV file"}

            
            if detected_delimiter in ["\t", " "]:  
                with open(Validation_file, "w", newline="", encoding="utf-8") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(headers)
                        csv_writer.writerows(data_rows)

                df = pd.read_csv(Validation_file)
            else:

                headers = data[0]  # Extract headers
                data_rows = data[1:]  # Extract data rows
                df = pd.DataFrame(data_rows, columns=headers)
                
                # Check if the first column contains only spaces or empty values (handling mixed data types)
                if df.iloc[:, 0].isna().all() or df.iloc[:, 0].str.strip().eq("").all():
                    df.drop(df.columns[0], axis=1, inplace=True)

                # Remove the last column if it contains only spaces or empty values
                if df.iloc[:, -1].isna().all() or df.iloc[:, -1].str.strip().eq("").all():
                    df.drop(df.columns[-1], axis=1, inplace=True)

            # Save the cleaned CSV file
            df.to_csv(Validation_file, index=False, encoding="utf-8")
        print("File saved successfully.")  # Debug
        return Validation_file

    except Exception as e:
        return {"error": str(e)} 

# Function to determine the type of a column dynamically
def determine_column_type(column_name):
    
    column_name = column_name.strip().lower()  # Normalize column name
    
    # Load feedback mapping from the database
    feedback_mapping = get_feedback_column_mapping()
    if column_name in feedback_mapping:
        return feedback_mapping[column_name]
    
    # Step 1: Direct match in user feedback
    if column_name in feedback_mapping:
        print(f"✅ Feedback match found: {column_name} -> {feedback_mapping[column_name]}")
        return feedback_mapping[column_name]
    
    # Keywords associated with different column types
    column_keywords = {
        "id": ["id", "identifier", "code", "number", "account", "reference", "unique"],
        "date": ["date", "dob", "birth", "purchase", "order", "expiry", "created"],
        "email": ["email", "mail"],
        "phone": ["phone", "mobile", "contact", "telephone"],
        "country": ["country", "nation", "region"],
        "age": ["age", "years", "user age"],
        "Name": ["name", "first name", "last name", "full name", "manager", "employee", "user", "customer", "client", "patient", "doctor", "student", "teacher", "professor", "engineer", "developer", "designer", "analyst", "consultant", "officer"],
        "address": ["address", "street", "city", "state"],
        "company_name": ["company", "organization", "business", "name"],
        "Amount": ["amount", "price", "cost", "total", "value", "paid amount"],
        "Salary": ["paid salary", "salary", "incentive", "payment"],
        "Transaction":["transaction", "transfer", "deal", "exchange", "trade", "withdrawal"]}

    column_name = column_name.lower().replace("_", " ").strip()
    doc = nlp(column_name)
    tokens = [token.text for token in doc]

    # Match against keywords for each type
    for col_type, keywords in column_keywords.items():
        for keyword in keywords:
            if keyword in tokens or keyword in column_name:
                return col_type
    

    # Fuzzy matching as another fallback
    close_matches = difflib.get_close_matches(column_name, feedback_mapping.keys(), n=1, cutoff=0.8)
    if close_matches:
        return feedback_mapping[close_matches[0]]

    # Fallback: Use regex for patterns commonly seen in certain column types
    if re.search(r"\b(id|code|number|transactionid|user_id)\b", column_name):
        return "id"
    elif re.search(r"\b(date)\b", column_name):
        return "date"
    elif re.search(r"\b(datetime)\b", column_name):
        return "date_time"
    elif re.search(r"\b(email|mail)\b", column_name):
        return "email"
    elif re.search(r"\b(phone|mobile|contact)\b", column_name):
        return "phone"
    elif re.search(r"\b(country|nation|region)\b", column_name):
        return "country"

    
    # If exact match is not found, use fuzzy matching
    close_matches = difflib.get_close_matches(column_name, feedback_mapping.keys(), n=1, cutoff=0.8)
    
    if close_matches:
        matched_column = close_matches[0]
        print(f"Fuzzy match found: {matched_column}")
        return feedback_mapping[matched_column]
    
    # Default to unknown if no matches
    return "Others"

# Fetch feedback data from the Feedback_table and create a lookup dictionary
def get_feedback_column_mapping():
    db = SessionLocal()
    feedback_data = db.query(Feedback).all()
    db.close()
    
    feedback_mapping = {}
    for entry in feedback_data:
        feedback_mapping[entry.Column_name.lower().strip()] = entry.Change_type.strip()
    
    return feedback_mapping

# Function to validate data using a generated regex pattern
def validate_data(data, pattern):
    regex = re.compile(pattern)
    return [value for value in data if not regex.fullmatch(str(value))]

def generate_regex_pattern(data):
    patterns = []

    for value in data:
        temp_pattern = ""
        value = str(value)  # Convert all values to strings for pattern generation

        for char in value:
            if char.isupper():
                temp_pattern += "[A-Z]"  # Match any uppercase letter
            elif char.islower():
                temp_pattern += "[a-z]"  # Match any lowercase letter
            elif char.isdigit():
                temp_pattern += "[0-9]"  # Match any digit
            elif not char.isalnum():
                temp_pattern += re.escape(char)  # Match any symbol or special character

        patterns.append(temp_pattern)

    if patterns:
        return max(set(patterns), key=patterns.count)  # Choose the most common pattern

    return " "  # Return a blank string if no patterns are generated

# Function to detect anomalies using IQR
def detect_anomalies_iqr(data):
    # Calculate the IQR for anomaly detection
    Q1 = data.quantile(0.05)
    Q3 = data.quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = data[(data < lower_bound) | (data > upper_bound)]
    return anomalies.tolist()

# Function to handle numeric data for Amount and Salary
def handle_numeric_column(data: pd.Series, column_name: str) -> List[float]:
    anomalies = []

    # Replace NaNs with "0" and ensure data is treated as strings for preprocessing
    for value in data.fillna("0").astype(str):
        # Check for invalid symbols (anything other than digits, comma, dot, minus, and currency symbols)
        if re.search(r'[^0-9\.,\-\₹\$\€]', value):  # Match any character that is not allowed
            anomalies.append(value)  # Append invalid value as anomaly
        else:
            # Only allow digits, comma, dot, minus, and currency symbols
            value_cleaned = re.sub(r'[^0-9\.,\-\₹\$\€]', '', value)  # Remove invalid symbols but keep the value

            # Convert cleaned value to numeric (if possible)
            value_cleaned = pd.to_numeric(value_cleaned, errors='coerce')

            if pd.notnull(value_cleaned):
                # Detect negative values
                if value_cleaned < 0:
                    anomalies.append(value)

                # Detect anomalies in non-negative values using IQR
                if value_cleaned >= 0:
                    positive_data = pd.Series([value_cleaned])
                    anomalies.extend(detect_anomalies_iqr(positive_data))

    return anomalies

# Function to handle numeric data for transaction
def handel_transactions_column(data: pd.Series, column_name: str) -> List[float]:
    anomalies = []

    # Replace NaNs with "0" and ensure data is treated as strings for preprocessing
    for value in data.fillna("0").astype(str):
        # Check for invalid symbols (anything other than digits, comma, dot, minus, and currency symbols)
        if re.search(r'[^0-9\.,\-\₹\$\€]', value):  # Match any character that is not allowed
            anomalies.append(value)  # Append invalid value as anomaly
        else:
            # Only allow digits, comma, dot, minus, and currency symbols
            value_cleaned = re.sub(r'[^0-9\.,\-\₹\$\€]', '', value)  # Remove invalid symbols but keep the value

            # Convert cleaned value to numeric (if possible)
            value_cleaned = pd.to_numeric(value_cleaned, errors='coerce')

            if pd.notnull(value_cleaned):
                # Detect anomalies in non-negative values using IQR
                if value_cleaned >= 0:
                    positive_data = pd.Series([value_cleaned])
                    anomalies.extend(detect_anomalies_iqr(positive_data))

    return anomalies

# Function to preprocess name (remove or handle specific symbols)
def preprocess_name(name: str) -> str:
    # Remove or replace undesired symbols like underscore (_) or hyphen (-)
    name = name.replace('_', ' ').replace('-', ' ')
    return name

def is_meaningful_text(text):
    # Simple NLP check to see if the text is meaningful (can be enhanced further)
    doc = nlp(text)
    # Check if the text is not empty, and contains at least one non-punctuation word
    return len(doc) > 0 and any([token.is_alpha for token in doc])

def detect_anomalies_categorical(data: pd.Series) -> List[int]:
    invalid_values = []

    for name in data.dropna().astype(str):
        # Preprocess name by removing undesired characters
        cleaned_name = preprocess_name(name)

        # Check if the name contains any numeric value
        if re.search(r'\d', cleaned_name):
            invalid_values.append(name)  # Append original name if it contains numeric value
        elif not re.match(r'^[a-zA-Z\s]+$', cleaned_name):  # Check if the name has only alphabets or spaces
            invalid_values.append(name)  # Append original name if it has symbols other than alphabets and spaces
    return invalid_values

# Function to validate data for each column type
def validate_column_data(column_name, column_data, column_type):
    # Define regex patterns for validation
    patterns = {
        "email": r"^(?!.*\.\.)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}$",
        "phone": r"^\+?[1-9]\d{0,2}\s?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$",
        "date": r"^(?:(?:\d{2}[-/.]\d{2}[-/.]\d{4})|(?:\d{4}[-/.]\d{2}[-/.]\d{2})|(?:\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}))(?:[ T]\d{2}:\d{2}(:\d{2})?)?$",
        "date_time":r"^\d{4}[-/]\d{2}[-/]\d{2}(?:[ T]\d{2}:\d{2}(:\d{2})?)?$",
        "country": r'^[A-Z][a-zA-Z\s-]+(?:[A-Z][a-zA-Z\s-]*)*$'}

    invalid_values = []

    # Check for company-related or address-related columns and skip validation
    company_keywords = ["company", "organization", "business", "firm"]
    address_keywords = ["address", "street", "city", "state", "location"]
    
    if any(keyword in column_name.lower() for keyword in company_keywords) or any(keyword in column_name.lower() for keyword in address_keywords):
        return invalid_values  # No validation, just return empty list

    if column_type in patterns:
        pattern = re.compile(patterns[column_type])
        for value in column_data.dropna():
            if not pattern.fullmatch(str(value)):
                invalid_values.append(value)

    elif column_type == "id":
        if pd.api.types.is_integer_dtype(column_data):
            # Validate integer IDs
            is_numeric = column_data.dropna().apply(lambda x: str(x).isdigit())
            if not is_numeric.all():
                invalid_values.extend([v for v in column_data[~is_numeric] if v not in invalid_values])

            numeric_data = pd.to_numeric(column_data, errors='coerce')
            invalid_values = []

            if not numeric_data.empty:
                differences = numeric_data.diff().dropna()
                
                # Check if at least 75% of the data follows a consecutive pattern
                if (differences == 1).sum() / len(differences) >= 0.50:
                    expected_value = numeric_data.iloc[0]
                    for actual_value in numeric_data:
                        if actual_value != expected_value:
                            invalid_values.append(actual_value)
                        expected_value += 1
                else:
                    # Handle random ID pattern
                    digit_lengths = numeric_data.astype(str).apply(len)
                    majority_length = digit_lengths.mode()[0]  # Most common digit length
                    invalid_values.extend(numeric_data[digit_lengths != majority_length].tolist())
        else:
            sample_data = column_data.dropna().astype(str).tolist()
            pattern = generate_regex_pattern(sample_data)
            print(pattern)
            invalid_data = validate_data(sample_data, pattern)
            invalid_values.extend([v for v in invalid_data if v not in invalid_values])

    elif column_type == "age":
        for value in column_data.dropna():
            if not (isinstance(value, (int, float)) and 0 <= value <= 100):
                invalid_values.append(value)

    elif column_type == "Name":
        invalid_values = detect_anomalies_categorical(column_data)

    elif column_type == "Amount" or column_type == "Salary":
      if any(re.search(r'[a-zA-Z]+[-_]+[a-zA-Z]*', str(value)) for value in column_data.dropna().astype(str)):
        # If regex finds non-numeric values, check if they are meaningful text
        for value in column_data.dropna().astype(str):
            if not is_meaningful_text(value):
                invalid_values.append(value)
      else:
        anomalies = handle_numeric_column(column_data, column_name)
        invalid_values.extend(anomalies)

    elif column_type == "Transaction":
      if any(re.search(r'[a-zA-Z]+[-_]+[a-zA-Z]*', str(value)) for value in column_data.dropna().astype(str)):
        # If regex finds non-numeric values, check if they are meaningful text
        for value in column_data.dropna().astype(str):
            if not is_meaningful_text(value):
                invalid_values.append(value)

      else:
        anomalies = handel_transactions_column(column_data, column_name)
        invalid_values.extend(anomalies)

    elif column_type == "Others":
        if any(re.search(r'^[a-zA-Z]+([-_ ]?[a-zA-Z]+)*$', str(value)) for value in column_data.dropna().astype(str)):
            for value in column_data.dropna().astype(str):
                if not is_meaningful_text(value):
                    invalid_values.append(value)
        else:
            column_data_str = column_data.astype(str)

            if column_data_str.str.isnumeric().all():
                column_data_numeric = pd.to_numeric(column_data_str)
                invalid_values.extend(detect_anomalies_iqr(column_data_numeric))
            elif pd.api.types.is_float_dtype(column_data):
                anomalies = detect_anomalies_iqr(column_data)
                invalid_values.extend(anomalies)
            else:
                sample_data = column_data.dropna().astype(str).tolist()
                converted_list = []
                for x in sample_data:
                    try:
                        # Attempt to convert to int or float
                        converted_value = int(x) if x.isdigit() else float(x)
                        converted_list.append(converted_value)
                    except ValueError:
                        # If conversion fails, append the value to invalid_values
                        invalid_values.append(x)

                # Detect anomalies in the successfully converted list
                invalid_values.extend(detect_anomalies_iqr(pd.Series(converted_list)))

    return invalid_values

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...), data_info: str = Form(...), sheet: str = Form(None)):
    global LATEST_FILE_PATH

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the file first
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        LATEST_FILE_PATH = file_path  # Update the global variable

        ext = os.path.splitext(file.filename)[1].lower()
        detected_encoding = None

        if ext in [".xlsx", ".xls"] and not sheet:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names  # Get all sheet names
            return JSONResponse(content={"message": "File processed successfully", "sheets": sheet_names})

        if ext in [".csv", ".txt"]:

            # Ensure encoding detection works
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read(10000))  
                detected_encoding = result['encoding']
            
            if data_info.lower() == "yes":
                text_data = "output_text.txt"
                try:
                    with open(file_path , 'r', encoding='utf-8', newline='') as csv_file, open(text_data, 'a', encoding='utf-8', newline='') as txt_file:
                        reader = csv.reader(csv_file)
                        for i, row in enumerate(reader):
                            if i < 5:  # Limit to first 5 rows
                                txt_file.write(','.join(row) + '\n')
                            else:
                                break  # Stop after 5 rows
                except FileNotFoundError:
                    print(f"Error: {file_path} not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
                flag=0
                    
                with open(text_data, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    first_row = next(reader, [])  # Read the first row safely
                    print(first_row)
                    lines = infile.readlines()
                    pattern = r'"'  # Regex pattern to detect double quotes

                    if re.search(pattern, ','.join(first_row)):
                        def detect_separator(first_row):
                            headers = [header.strip('"') for header in first_row]
                            match = re.findall(r'"[^"]*"(.*?)"[^"]*"', first_row)
                            separators = set(match)
                            separators.discard('')  # Remove empty matches
                            if len(separators) == 1:
                                return separators.pop()
                            else:
                                raise ValueError("Could not detect a unique separator.")
                        detected_delimiter = detect_separator(','.join(first_row))
                        print(f"Detected delimiter: {detected_delimiter}")
                        flag=1
                        if not detected_delimiter:
                            raise ValueError("Could not detect the separator.")
                        
                    else:
                        detected_delimiter= symbol_delimiter(lines)
                        print(f"Special character detected: {detected_delimiter}")
                        
                # Clear output_text.txt after processing the current file
                with open("output_text.txt", "w", encoding="utf-8") as outfile:
                    outfile.write("")
        
                if not detected_delimiter:
                    raise ValueError("No valid delimiter found in the file.")
                
                # Read cleaned CSV using Pandas
                Validation_file = Corrected_file(file_path, detected_delimiter, flag)
                data = pd.read_csv(Validation_file)
                            
            else:
                # Read CSV directly
                data = pd.read_csv(file_path, encoding=detected_encoding)

        elif ext == ".xlsx":
            if not sheet:
                return JSONResponse(content={"error": "Sheet name is required for Excel files."}, status_code=400)
            
            data = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")

        elif ext == ".xls":
            if not sheet:
                return JSONResponse(content={"error": "Sheet name is required for Excel files."}, status_code=400)
            
            try:
                # Try using xlrd for old .xls files
                data = pd.read_excel(file_path, sheet_name=sheet, engine="xlrd")
            except Exception as e:
                print(f"⚠️ xlrd failed: {e}")
                
                try:
                    # Try using openpyxl for newer formats
                    data = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
                except Exception as e:
                    print(f"⚠️ openpyxl failed: {e}")
                    return JSONResponse(content={"error": "Failed to read .xls file. Ensure it is in a supported format."}, status_code=400)
        else:
            return JSONResponse(content={"error": "Unsupported file type."}, status_code=400)

        results = {}
        for column in data.columns:
            column_data = data[column]
            column_type = determine_column_type(column)
            column_dtype = column_data.dtype
            # Ensure we check for anomalies first
            invalid_values = validate_column_data(column, column_data, column_type)
            invalid_values_with_index = [
                {"Row Number": index + 1, "Value": value}
                for index, value in column_data[column_data.isin(invalid_values)].items()
            ]
            if len(invalid_values) == len(column_data):
                # If all values are anomalies, don't list them
                results[column] = {
                    "Name of Column": column,
                    "Type": column_type,
                    "Data Type": str(column_dtype),
                    "Anomaly Values": [],
                    "Message": "The data format in the Column is invalid"
                }
            else:
                results[column] = {
                    "Name of Column": column,
                    "Type": column_type,
                    "Data Type": str(column_dtype),
                    "Anomaly Values": invalid_values_with_index,
                    "Message": f"{len(invalid_values)} anomaly detected" if invalid_values else "No anomaly detected"
                }
   
        # Fine-tune the model after processing the file
        fine_tune_model()
        return JSONResponse(content={"message": "File processed successfully", "results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Define column keywords
def load_column_keywords():
    return {
        "id": ["id", "identifier", "code", "number", "account", "reference", "unique"],
        "date": ["date", "dob", "birth", "purchase", "order", "expiry", "created"],
        "date_time": ["date", "time", "timestamp", "datetime"],
        "email": ["email", "mail"],
        "phone": ["phone", "mobile", "contact", "telephone"],
        "country": ["country", "nation", "region"],
        "age": ["age", "years", "user age"],
        "Name": ["name", "first name", "last name", "full name", "Sales Man","Manager", "Employee", "User", "Customer", "Client", "Patient", "Doctor", "Student", "Teacher", "Professor", "Engineer", "Developer", "Designer", "Analyst", "Consultant", "Officer"],
        "address": ["address", "street", "city", "state"],
        "company_name": ["company", "organization", "business", "name"],
        "Amount": ["amount", "price", "cost", "total", "value", "paid amount"],
        "Salary": ["paid salary", "salary", "incentive", "payment"],
        "Transaction": ["transaction", "transfer", "deal", "exchange", "trade", "withdrawal"],
    }

# Utility function to clean input by removing problematic characters
def clean_input(input_string):
    # Remove unwanted special characters (anything that isn't normal printable text)
    return re.sub(r'[^\x00-\x7F]+', '', input_string)

# Feedback endpoint
@app.post("/feedback/")
async def save_feedback(request: Request):
    try:
        db = SessionLocal()
        raw_json = await request.json()
        
        # Ensure "feedback_data" exists and is a string before proceeding
        feedback_data = raw_json.get("feedback_data")
        
        if not feedback_data:
            return JSONResponse(content={"error": "No feedback data provided."}, status_code=400)
        
        try:
            feedback_list = json.loads(feedback_data) if isinstance(feedback_data, str) else feedback_data
        except json.JSONDecodeError as e:
            return JSONResponse(content={"error": f"Invalid JSON format: {str(e)}"}, status_code=400)

        if not isinstance(feedback_list, list):
            return JSONResponse(content={"error": "Feedback data must be a list of dictionaries."}, status_code=400)

        if not LATEST_FILE_PATH or not os.path.exists(LATEST_FILE_PATH):
            return JSONResponse(content={"error": "No uploaded data file found."}, status_code=400)

        # Retrieve the "Delimited" field value from the request
        delimited_value = raw_json.get("Delimited", "no")  # ✅ Ensure we get a value
               
        # Detect the file extension and read accordingly
        file_extension = os.path.splitext(LATEST_FILE_PATH)[1].lower()

        try:
            if file_extension in ['.xlsx', '.xls']:
                # Handle Excel files
                data = pd.read_excel(LATEST_FILE_PATH)
            elif file_extension == '.csv':
                # Handle CSV files
                # Choose the file based on "Delimited" value
                if delimited_value == "yes":
                    file_path = os.path.join("C:\ZingMind_December2024\Regexify", "Corrected_file.csv") # Use Validation_file.csv
                    data = pd.read_csv(file_path)
                    print(data.head())
                else:
                    if not LATEST_FILE_PATH or not os.path.exists(LATEST_FILE_PATH):
                        return JSONResponse(content={"error": "No uploaded data file found."}, status_code=400)
                    file_path = LATEST_FILE_PATH  # Use the uploaded file
                    data = pd.read_csv(LATEST_FILE_PATH)
            else:
                # Return error if file is neither .csv nor .xlsx/.xls
                return JSONResponse(content={"error": "Unsupported file format. Please upload a .csv or .xlsx/.xls file."}, status_code=400)
            
            available_columns = data.columns.str.lower().tolist()
        except Exception as e:
            return JSONResponse(content={"error": f"Failed to read the file: {str(e)}"}, status_code=500)
        
        feedback_objects = []  # ✅ Collect Feedback objects
        
        # Validate and save each feedback entry
        for feedback in feedback_list:
            if not isinstance(feedback, dict):
                return JSONResponse(content={"error": "Each feedback entry must be a dictionary."}, status_code=400)
            
            column_name = clean_input(feedback.get("column_name", ""))          
            change_type = clean_input(feedback.get("change_type", ""))
            feedback_text = clean_input(feedback.get("feedback_text", ""))

            # Decode each field in case there are hidden problematic characters
            try:
                column_name = column_name.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                change_type = change_type.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                feedback_text = feedback_text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            except UnicodeDecodeError as e:
                print(f"❌ Error decoding field: {str(e)}")

            # Clean and normalize the available columns
            available_columns = [col.strip().lower() for col in data.columns.tolist()]
            column_name = column_name.strip().lower()  

            if column_name in available_columns:
                matched_column = column_name  
            else:
                close_matches = difflib.get_close_matches(column_name, available_columns, n=1, cutoff=0.8)
                if close_matches:
                    matched_column = close_matches[0]
                else:
                    return JSONResponse(
                        content={"error": f"Invalid column name: {feedback['column_name']}. Available columns: {data.columns.tolist()}"},
                        status_code=400,
                    )
                 # Use the closest match for the column
                matched_column = available_columns[cleaned_columns.index(close_matches[0])]

            if not change_type or not feedback_text:
                return JSONResponse(
                    content={"error": "Change type and feedback text cannot be empty."},
                    status_code=400,
                )

            # Save feedback to the database
            new_feedback = Feedback(
                Column_name=matched_column,
                Change_type=change_type,
                feedback_text=feedback_text,
            )
            db.add(new_feedback)
            feedback_objects.append(new_feedback)
            
        db.commit()
         # ✅ Refresh objects so they can be used after session closes
        for feedback_obj in feedback_objects:
            db.refresh(feedback_obj)
        db.close()

        # Fine-tune the model
        fine_tune_model(feedback_objects)
        
        return JSONResponse(content={"message": "Feedback saved and model fine-tuned successfully."})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Function to fine-tune the model using feedback from the Feedback_table
def fine_tune_model(feedback_dataset=None):
    try:
        db = SessionLocal()
        # Use provided dataset or fetch from database
        if feedback_dataset is None:
            feedback_dataset = db.query(Feedback).all()
            db.close()

        if not feedback_dataset:
            print("No feedback entries found. Skipping fine-tuning.")
            return

        # Iterate over feedback entries
        for entry in feedback_dataset:
            column_name = entry.Column_name.lower().strip()
            change_type = entry.Change_type.strip()
            feedback_text = entry.feedback_text.strip()
            
            # Convert bytes to strings (only if needed)
            if isinstance(column_name, bytes):
                column_name = column_name.decode('utf-8')
            if isinstance(change_type, bytes):
                change_type = change_type.decode('utf-8')
            if isinstance(feedback_text, bytes):
                feedback_text = feedback_text.decode('utf-8')

            column_name = column_name.lower().strip()
            change_type = change_type.strip()
            feedback_text = feedback_text.strip()

        feedback_mapping = {}
        for entry in feedback_dataset:  # Use feedback_dataset instead of feedback_data
            feedback_mapping[entry.Column_name.lower()] = entry.Change_type

        return feedback_mapping

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")

# Serve HTML frontend
@app.get("/", response_class=HTMLResponse)

async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

