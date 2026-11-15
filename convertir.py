import pandas as pd

try:
    print("1. Intentando leer y convertir nodos.csv...")
    df_n = pd.read_csv("nodos.csv")
    df_n.to_parquet("nodos.parquet", index=False)
    print("-> ¡nodos.parquet creado con éxito!")

    print("2. Intentando leer y convertir aristas.csv...")
    df_a = pd.read_csv("aristas.csv")
    df_a.to_parquet("aristas.parquet", index=False)
    print("-> ¡aristas.parquet creado con éxito!")

    print("3. Intentando leer y convertir dataset_ccxt_completo.csv...")
    df_c = pd.read_csv("dataset_ccxt_completo.csv")
    df_c.to_parquet("dataset_ccxt_completo.parquet", index=False)
    print("-> ¡dataset_ccxt_completo.parquet creado con éxito!")

    print("\n🎉 ¡Proceso finalizado sin errores!")

except Exception as e:
    print(f"\n❌ Ocurrió un error: {e}")