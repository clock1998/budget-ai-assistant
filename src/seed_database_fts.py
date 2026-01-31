import sqlite3
import pandas as pd

db = sqlite3.connect("my_db.db")


db.execute("""    
    DELETE VIRTUAL TABLE IF EXISTS fts_documents;
""")

db.execute("""
    CREATE VIRTUAL TABLE fts_documents USING 
               fts5(
                business_name,
                business_domain, 
                business_niche_description,
                tokenize = 'trigram'
               );
""")

businesses = pd.read_csv('data.csv')
data = [];
for i, row in businesses.iterrows():
    # Insert text into documents
    data.append((row['business_name'], row['business_domain'], row['business_niche_description']));

db.executemany('''INSERT INTO fts_documents (
                   business_name, 
                   business_domain, 
                   business_niche_description) VALUES (?, ?, ?)'''
                   , data);    
db.commit()
db.close()

