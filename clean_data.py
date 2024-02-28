"""Taller evaluable presencial"""

import pandas as pd


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""

    #
    # Esta parte es igual al taller anterior
    #
    df = pd.read_csv(input_file)
    return df


def create_key(df, n):
    """Cree una nueva columna en el DataFrame que contenga el key de la columna 'text'"""

    df = df.copy()
    df["key"] = (
        df["text"]
        .str.strip()
        .str.lower()
        .str.replace("-", "")
        .str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        .str.split()
    )

    # ------------------------------------------------------
    # Esta es la parte especifica del algoritmo de n-gram:

    df["key"] = (
        df["key"]
        #
        # - Una el texto sin espacios en blanco
        .str.join("")
        #
        # - Convierta el texto a una lista de n-gramas
        .map(lambda x: [x[t : t + n - 1] for t in range(len(x))])
        #
        # - Ordene la lista de n-gramas y remueve duplicados
        .apply(lambda x: sorted(set(x)))
        #
        # - Convierta la lista de ngramas a una cadena
        .str.join("")
    )
    ## ------------------------------------------------------

    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    #
    # Este código es identico al anteior
    #
    df = df.copy()
    df = df.sort_values(by=["key", "text"], ascending=[True, True])
    keys = df.drop_duplicates(subset="key", keep="first")
    key_dict = dict(zip(keys["key"], keys["text"]))
    df["cleaned"] = df["key"].map(key_dict)

    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    #
    # Este código es identico al anteior
    #
    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "text"})
    df.to_csv(output_file, index=False)


def main(input_file, output_file, n=2):
    """Ejecuta la limpieza de datos"""
    #
    # Este código es identico al anteior
    #
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
