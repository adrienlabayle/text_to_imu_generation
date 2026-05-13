import pandas as pd
from pathlib import Path

# 1. Configuration des chemins
ROOT = Path("D:/IMDS/MALMO/IMU_LM_Data")
input_path = ROOT / "data" / "merged_dataset" / "unified_dataset.parquet"
output_path = ROOT / "data" / "merged_dataset" / "filtered_activities_dataset.parquet"

# 2. Liste des activités à conserver
# Ces labels correspondent aux noms canoniques utilisés dans tes processus de merge
target_activities = ["posture_stationary", "walk", "rest_inactive"]

print(f"Chargement et filtrage de : {input_path.name}...")

# 3. Lecture et filtrage
# On utilise .isin() qui est très rapide pour filtrer sur une liste de valeurs
df = pd.read_parquet(input_path)
df_filtered = df[df['global_activity_label'].isin(target_activities)].copy()

# 4. Affichage des résultats pour vérification
print(f"\nFiltrage terminé :")
print(f" - Lignes avant : {len(df):,}")
print(f" - Lignes après : {len(df_filtered):,}")
print(f" - Réduction de taille : {100 * (1 - len(df_filtered)/len(df)):.1f}%")

# Vérification des éléments distincts restants (doit afficher tes 3 labels)
print(f" - Activités conservées : {df_filtered['global_activity_label'].unique()}")

# 5. Sauvegarde du nouveau fichier plus léger
df_filtered.to_parquet(output_path, index=False)
print(f"\n[SUCCÈS] Nouveau dataset sauvegardé sous : {output_path.name}")