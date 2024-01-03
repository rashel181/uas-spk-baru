import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance as d

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

if 'nilai_kriteria' not in st.session_state:
    st.session_state.nilai_kriteria = np.array([])

# Label kriteria
label = ['benefit', 'benefit', 'cost', 'cost']

# Bobot kriteria
bobot = np.array([0.25, 0.3, 0.2, 0.25])

# Alternatif
alternatif = [
    'Pembelian Mesin Baru',
    'Pelatihan Karyawan',
    'Penggunaan Energi Terbarukan',
    'Ekspansi Pabrik'
]

def click_button():
    st.session_state.clicked = True
    
def normalization(matrix):
    matrix = matrix.transpose() # tukar baris & kolom sehingga baris = kriteria, kolom = alternatif
    normMatrix = []

    for i in range(matrix.shape[0]): # Looping per baris (kriteria)
        # List normalisasi per kriteria
        rowValues = []

        # Menghitung total nilai alternatif yang dikuadratkan pada tiap kriteria
        sumRow = sum([pow(x,2) for x in matrix[i]])

        for j in range(matrix[i].shape[0]): # Looping per kolom (alternatif)
            # Membagi nilai alternatif asli dengan hasil akar total nilai alternatif kuadrat pada tiap kriteria
            r_ij = matrix[i][j] / math.sqrt(sumRow)

            # Masukkan hasil normalisasi ke list tiap baris
            rowValues.append(r_ij)

        #Masukkan hasil normalisasi nilai alternatif per kriteria ke matriks hasil normalisasi
        normMatrix.append(rowValues)

    # Ubah ke dalam bentuk numpy array
    normMatrix = np.asarray(normMatrix)

    # Return dalam bentuk transpose agar kembali ke format awal
    return normMatrix.transpose()

# Prosedur optimalisasi nilai alternatif / atribut
def optimalization(normalizedMatrix, criteriaWeights):
    # Buat salinan nilai ternormalisasi dan transpose
    optimizedMatrix = normalizedMatrix.transpose()

    for i in range(criteriaWeights.shape[0]): # Looping tiap kriteria
        # Kalkulasi nilai optimal tiap nilai alternatif pada masing-masing kriteria
        optimizedMatrix[i] = [r * criteriaWeights[i] for r in optimizedMatrix[i]]

    # Ubah ke bentuk numpy array
    optimizedMatrix = np.asarray(optimizedMatrix)

    # Return dalam bentuk transpose agar kembali ke format awal
    return optimizedMatrix.transpose()

def ideal(w_norm, c_label):
    w_norm = w_norm.transpose()
    a_positif = []
    a_negatif = []

    for i in range(w_norm.shape[0]):
        if c_label[i] == 'benefit':
            # Untuk ideal positif
            a_max = max(w_norm[i])
            a_positif.append(a_max)

            # Untuk ideal negatif
            a_min = min(w_norm[i])
            a_negatif.append(a_min)
        elif c_label[i] == 'cost':
            # Untuk ideal positif
            a_max = min(w_norm[i])
            a_positif.append(a_max)

            # Untuk ideal negatif
            a_min = max(w_norm[i])
            a_negatif.append(a_min)

    ideal_value = np.array([a_positif, a_negatif])

    return ideal_value

def alt_ideal_distance(w_norm, ideal_v):
    w_norm = w_norm.transpose()
    ideal_v = ideal_v.transpose()

    d_positif = []
    d_negatif = []

    # Kalkulasi Jarak
    for i in range(w_norm[0].shape[0]):
        # positif
        dp = d.euclidean(w_norm[:,i], ideal_v[:,0])
        d_positif.append(dp)

        # negatif
        dn = d.euclidean(ideal_v[:,1], w_norm[:,i])
        d_negatif.append(dn)

    d_positif = np.asarray(d_positif)
    d_negatif = np.asarray(d_negatif)

    d_value = np.array([d_positif, d_negatif])

    return d_value.transpose()

def final_rank(distance, alternatif_names):
    v = []

    for i in range(distance.shape[0]):
        vi = distance[i][1] / (distance[i][1] + distance[i][0])
        v.append((alternatif_names[i], vi))
        
    sorted_rankings = sorted(v, key=lambda x: x[1], reverse=True)

    return sorted_rankings

def simpanData(c1, c2, c3, c4):
    if 'nilai_kriteria' not in st.session_state:
        st.session_state.nilai_kriteria = np.array([[c1, c2, c3, c4]])
    else:
        dataLama = st.session_state.nilai_kriteria
        if dataLama.size == 0:  # Check if the array is empty
            dataLama = np.empty((0, 4))  # Initialize with an empty array
        dataBaru = np.vstack([dataLama, [c1, c2, c3, c4]])
        st.session_state.nilai_kriteria = dataBaru


def run():
    st.set_page_config(
        page_title="IMPLEMENTASI METODE TOPSIS | UAS",
        page_icon="🤫🧏😾🗿",
    )
    
    st.write("# Implementasi Metode TOPSIS")
    st.markdown(
        """
        ### CONTOH KASUS :
        
        PT. XYZ adalah perusahaan manufaktur yang berencana melakukan ekspansi produksi. Beberapa alternatif investasi akan dievaluasi untuk menentukan alternatif terbaik. Evaluasi dilakukan berdasarkan 4 kriteria dengan bobot tertentu.
        
        Kriteria:
        - Ketersediaan Sumber Daya Manusia (SDM) (1 = kurang, 2 = cukup, 3 = baik, 4 = sangat baik)
        Jenis: Benefit (Manfaat)
        Bobot: 25%
        - Teknologi Produksi Terbaru (1 = tidak ada, 2 = ada tetapi kurang canggih, 3 = canggih, 4 = sangat canggih)
        Jenis: Benefit (Manfaat)
        Bobot: 30%
        - Biaya Investasi Awal (dalam jutaan Rupiah) 
        Jenis: Cost (Biaya)
        Bobot: 20%
        - Dampak Lingkungan (1 = tinggi, 2 = sedang, 3 = rendah, 4 = sangat rendah)
        Jenis: Cost (Biaya)
        Bobot: 25%
        
        Alternatif Investasi:
        - A1: Pembelian Mesin Baru
        - A2: Pelatihan Karyawan
        - A3: Penggunaan Energi Terbarukan
        - A4: Ekspansi Pabrik
        """
    )
    
    st.divider()
    
    st.write("## INPUT NILAI KRITERIA UNTUK TIAP ALTERNATIF (LAKUKAN HINGGA 4 KALI PENGISIAN UNTUK TIAP ALTERNATIF)")
    
    c1 = st.number_input("Nilai C1", min_value=1, max_value=4, value=1, step=1)
    c2 = st.number_input("Nilai C2", min_value=1, max_value=4, value=1, step=1)
    c3 = st.slider("Nilai C3 (JUTA)", min_value=10, max_value=1000, value=10, step=10)
    c4 = st.number_input("Nilai C4", min_value=1, max_value=4, value=1, step=1)
    
    if st.button("Simpan", type='primary', on_click=click_button):
        simpanData(c1,c2,c3,c4)
        
    if st.session_state.clicked:
        data = st.session_state.nilai_kriteria
        df = pd.DataFrame(data, columns=('C1','C2','C3','C4'))
        st.dataframe(df)

        if st.button("Proses"):
            prosesData()
            
      
def prosesData():
    P = st.session_state.nilai_kriteria
    
    # NORMALISASI
    norm_x = normalization(P)
    st.write("## Normalisasi:")
    st.text(norm_x)
    
    # OPTIMISASI
    criteria_weights = bobot
    weighted_norm_x = optimalization(norm_x, criteria_weights)
    st.write("## OPTIMALISASI:")
    st.text(weighted_norm_x)
    
    # IDEAL
    ideal_values = ideal(weighted_norm_x, label)
    st.write("## Ideal Values:")
    st.text(ideal_values)
    
    # ALT IDEAL DISTANCE
    separation_distance = alt_ideal_distance(weighted_norm_x, ideal_values)
    st.write("## ALT IDEAL DISTANCE:")
    st.text(separation_distance)
    
    # FINAL RANKINGS
    rankings = final_rank(separation_distance, alternatif)
    st.write("## FINAL RANKINGS:")
    for rank, (alternatif_name, value) in enumerate(rankings, start=1):
        st.write(f"{rank}. Alternatif: {alternatif_name}, Ranking Value: {value}")
    
if __name__ == "__main__":
    run()