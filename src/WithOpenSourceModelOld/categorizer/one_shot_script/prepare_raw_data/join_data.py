import pandas as pd

# 1. Load the datasets
# Note: In a real scenario, these would be file paths like 'entreprise.csv'
entreprise_df = pd.read_csv('Entreprise.csv', 
                            usecols=['NEQ', 'COD_ACT_ECON_CAE', 'DESC_ACT_ECON_ASSUJ'],
                            dtype={'NEQ': str, 'COD_ACT_ECON_CAE': str}) # DESC_ACT_ECON_ASSUJ for business niche.
nom_df = pd.read_csv('Nom.csv', 
                     usecols=['NEQ','STAT_NOM', 'NOM_ASSUJ','NOM_ASSUJ_LANG_ETRNG','TYP_NOM_ASSUJ'],
                     dtype={'NEQ': str, 'STAT_NOM': str})
domaine_valeur_df = pd.read_csv('DomaineValeur.csv', 
                                dtype={'COD_DOM_VAL': str})

# 2. Filter for 'Active' names only (optional but recommended) Filter TYP_NOM_ASSUJ for Doing Business As names
# This prevents one company from appearing 10 times because of old name history
# active_noms = nom_df[(nom_df['STAT_NOM'] == 'A') ]

# 3. Perform the Joins
# Join Entreprise with its Active Name
entrepise_with_active_names = pd.merge(entreprise_df, nom_df, on='NEQ', how='left')

# Create a specific dictionary just for Economic Activities
cae_dictionary = domaine_valeur_df[domaine_valeur_df['TYP_DOM_VAL'] == 'ACT_ECON']

# 4. Join with domaine values.

entreprise_decoded = pd.merge(
    entrepise_with_active_names, 
    cae_dictionary[['COD_DOM_VAL', 'VAL_DOM_FRAN']], 
    left_on='COD_ACT_ECON_CAE', 
    right_on='COD_DOM_VAL', 
    how='left'
)

# 5. Save or View the result
final_output = entreprise_decoded[[
    'NEQ',
    'NOM_ASSUJ', 
    'VAL_DOM_FRAN', 
    'DESC_ACT_ECON_ASSUJ'
]].rename(columns={
    'NEQ': 'Id',
    'NOM_ASSUJ': 'business_name',
    'VAL_DOM_FRAN': 'business_domain',
    'DESC_ACT_ECON_ASSUJ': 'business_niche_description'
})
print(final_output.head())
final_output.to_csv('data.csv',index=False)