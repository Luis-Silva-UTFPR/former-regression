# import geopandas as gpd
# import os

# def filter_valid_npz_files(gdf, year_range, output_filepath):
#     """
#     Filtra as linhas do GeoDataFrame que possuem arquivos .npz válidos e salva em um novo arquivo parquet.
    
#     Parameters:
#         gdf (GeoDataFrame): O GeoDataFrame contendo as colunas 'downloaded_filepath_*'.
#         year_range (list): Lista de anos para verificar as colunas 'downloaded_filepath_*'.
#         output_filepath (str): Caminho para salvar o GeoDataFrame com arquivos válidos.
        
#     Returns:
#         GeoDataFrame: O GeoDataFrame contendo apenas as linhas com arquivos válidos.
#     """
#     valid_rows = []

#     # Itera sobre os anos para verificar os caminhos dos arquivos
#     for year in year_range:
#         file_column = f"downloaded_filepath_{year}"
        
#         # Verifica cada linha no GeoDataFrame
#         for idx, file_path in enumerate(gdf[file_column]):
#             if file_path and os.path.isfile(file_path) and file_path.endswith('.npz'):
#                 valid_rows.append(idx)

#     # Filtra o GeoDataFrame para manter apenas as linhas com arquivos válidos
#     valid_gdf = gdf.loc[valid_rows]
    
#     # Salva o GeoDataFrame com arquivos válidos em um novo arquivo Parquet
#     valid_gdf.to_parquet(output_filepath)

#     return valid_gdf

# # Exemplo de uso
# if __name__ == "__main__":
#     # Carregar o GeoDataFrame (substitua pelo caminho real do seu arquivo)
#     gdf = gpd.read_parquet('results_pretrain.parquet')
    
#     # Definir a faixa de anos que você deseja verificar
#     year_range = list(range(2018, 2022))  # Exemplo para os anos de 2018 a 2023

#     # Caminho para salvar o arquivo filtrado com arquivos válidos
#     output_filepath = 'pretrain_valid.parquet'

#     # Filtrar as linhas válidas e gerar o novo arquivo Parquet
#     valid_gdf = filter_valid_npz_files(gdf, year_range, output_filepath)

#     print(f"GeoDataFrame filtrado com arquivos válidos foi salvo em: {output_filepath}")
#     print(f"Linhas válidas: {len(valid_gdf)}")

import os

# Executa o comando nvidia-smi para mostrar a utilização da GPU
os.system('nvidia-smi')
