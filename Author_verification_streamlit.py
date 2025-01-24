# Import required libraries
import pandas as pd
import PyPDF2
import openai
import requests
import re
import json
import streamlit as st

# Step 1: Download PDF from URL
def download_pdf_from_url(url, output_path="downloaded_file.pdf"):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        else:
            return f"Error downloading PDF: HTTP {response.status_code}"
    except Exception as e:
        return f"Error downloading PDF: {e}"

# Step 2: Extract text from the first page of PDF
def extract_first_page_text(file_path):
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            first_page_text = reader.pages[0].extract_text()
        return first_page_text
    except Exception as e:
        return f"Error extracting PDF: {e}"

# Step 3: Extract identifiers (DOI, HAL-Id, etc.) from text
def extract_identifiers(text):
    identifiers = {}
    # Extract DOI
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    identifiers['DOI'] = doi_match.group(0) if doi_match else "Not found"

    # Extract HAL-Id
    hal_id_match = re.search(r"hal-[a-zA-Z0-9]+", text, re.IGNORECASE)
    identifiers['HAL-Id'] = hal_id_match.group(0) if hal_id_match else "Not found"

    return identifiers

# Step 4: Identify authors and affiliations using OpenAI API
def identify_authors_and_affiliations_with_gpt(api_key, text):
    openai.api_key = api_key
    prompt = (
        "Extract the names of authors and their affiliations from the following text. "
        "Format the output as a JSON array of objects with 'Author' and 'Affiliation' as keys:\n\n" + text
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000
        )
        output = response['choices'][0]['message']['content'].strip()
        # Parse the JSON response
        return json.loads(output)
    except json.JSONDecodeError as json_err:
        return f"Error parsing JSON from OpenAI response: {json_err}"
    except Exception as e:
        return f"Error using OpenAI API: {e}"

# Streamlit App
st.title("PDF Author and Identifier Extractor")

# Input: URL of the PDF
pdf_url = st.text_input("Enter the URL of the PDF:")

# Input: OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if pdf_url and api_key:
    if st.button("Analyze PDF"):
        with st.spinner("Downloading and analyzing the PDF..."):
            # Download the PDF
            pdf_path = download_pdf_from_url(pdf_url)

            if not pdf_path.endswith(".pdf"):
                st.error(pdf_path)  # Display error if download failed
            else:
                # Extract text from the first page
                first_page_text = extract_first_page_text(pdf_path)

                if "Error" in first_page_text:
                    st.error(first_page_text)
                else:
                    # Extract identifiers
                    identifiers = extract_identifiers(first_page_text)
                    st.write("Identifiers extracted:", identifiers)

                    # Identify authors and affiliations using GPT
                    authors_and_affiliations = identify_authors_and_affiliations_with_gpt(api_key, first_page_text)

                    if isinstance(authors_and_affiliations, str) and "Error" in authors_and_affiliations:
                        st.error(authors_and_affiliations)
                    else:
                        # Create a DataFrame for authors and affiliations
                        df_authors = pd.DataFrame(authors_and_affiliations)

                        # Add identifiers to the DataFrame
                        df_authors["DOI"] = identifiers['DOI']
                        df_authors["HAL-Id"] = identifiers['HAL-Id']

                        # Display results in Streamlit
                        st.write("Extracted Authors and Affiliations:")
                        st.dataframe(df_authors)

                        # Download button for the CSV
                        csv = df_authors.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="author_and_identifiers_report.csv",
                            mime="text/csv"
                        )
