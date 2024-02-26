"""Taller evaluable presencial"""

import pandas as pd
import nltk


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""

    data = pd.read_csv(input_file, sep="\t")
    return data


def create_key(df, n):
    """Cree una nueva columna en el DataFrame que contenga el key de la columna 'text'"""

    new_df = df.copy()
    df = df.copy()

    """Con este paso, se aplica el algoritmo de Porter para encontrar la raíz de cada palabra"""
    # Copie la columna 'text' a la columna 'stem'
    df["stem"] = new_df["text"]
    df["stem"] = (
        df["stem"]
        # Remueva los espacios en blanco al principio y al final de la cadena
        .str.strip()
        # Convierta el texto a minúsculas
        .str.lower()
        # Transforme palabras que pueden (o no) contener guiones por su version sin guion.
        .str.replace("-", "")
        # Remueva puntuación y caracteres de control
        .str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        # Convierta el texto a una lista de tokens
        .str.split()
        # Transforme cada palabra con un stemmer de Porter
        .apply(lambda x: [nltk.PorterStemmer().stem(w) for w in x])
        # Ordene la lista de tokens y remueve duplicados
        .apply(lambda x: sorted(set(x)))
        # Convierta la lista de tokens a una cadena de texto separada por espacios
        .str.join(" ")
    )
    
    """Con este paso, se crean los n-gramas en la columna key"""
    # Copie la columna 'text' a la columna 'key'
    df["key"] = df["text"]
    df["key"] = (
        df["key"]
        # Remueva los espacios en blanco al principio y al final de la cadena
        .str.strip()
        # Convierta el texto a minúsculas
        .str.lower()
        # Transforme palabras que pueden (o no) contener guiones por su version sin guion.
        .str.replace("-", "")
        # Remueva puntuación y caracteres de control
        .str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        # Convierta el texto a una lista de tokens
        .str.split()
        # Una el texto sin espacios en blanco
        .str.join("")
        # Convierta el texto a una lista de n-gramas
        .map(lambda x: [x[i : i + n - 1] for i in range(len(x))])
        # Ordene la lista de n-gramas y remueve duplicados
        .apply(lambda x: sorted(set(x)))
        # Convierta la lista de ngramas a una cadena
        .str.join("")
    ) 

    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()
    
    # Ordene el dataframe por 'stem' y 'text'
    df = df.sort_values(by=["stem", "text"]).copy()
    # Seleccione la primera fila de cada grupo de 'stem'
    stem = df.groupby("stem").first().reset_index()
    # Cree un diccionario con 'stem' como clave y 'text' como valor
    stem = stem.set_index("stem")["text"].to_dict()

    # Cree la columna 'cleaned' usando el diccionario
    df["cleaned"] = df["stem"].map(stem)
    
    # La columna stem no se requiere en el archivo final, por lo que procede a ser eliminada
    df.drop("stem", axis=1, inplace=True)
    # Ordene el dataframe por 'key' y 'text'
    df = df.sort_values(by=["key", "text"])

    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""

    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "text"})
    df.to_csv(output_file, index=False)


def main(input_file, output_file, n=2):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_key(df, n)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
