# Laporan Proyek Machine Learning - Ivan Andrianto

## Domain Proyek

Risiko kredit terjadi ketika peminjam tidak dapat melakukan pembayaran reguler dan gagal memenuhi kewajibannya. Ini berarti bahwa pemberi pinjaman tidak akan dibayar untuk tepat waktu yang membuat arus kas pemberi pinjaman akan terganggu. Dalam skenario terburuk, pemberi pinjaman mungkin terpaksa membatalkan semua atau sebagian pinjaman, yang mengakibatkan kerugian yang sangat fatal. 

Sangat sulit dan kompleks untuk memprediksi kemungkinan seseorang gagal membayar utang. Pada saat yang sama, penilaian risiko kredit yang tepat dapat membantu membatasi risiko kerugian karena gagal bayar dan keterlambatan pembayaran. Sebagai imbalan untuk mengambil risiko kredit, pemberi pinjaman akan menerima bunga dari peminjam. Pemberi pinjaman atau investor akan mengenakan tingkat bunga yang lebih tinggi atau menolak untuk memberikan pinjaman jika risiko kredit lebih tinggi. Berangkat dari situasi yang kompleks tersebut, diperlukan sebuah model Machine Learning yang melakukan klasifikasi dengan tujuan membantu dalam pengambilan keputusan untuk menyetujui atau tidaknya sebuah peminjaman.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang sudah dipaparkan sebelumnya, berikut rincian masalah yang dapat diselesaikan dalam proyek ini:
- Apa saja faktor yang paling memengaruhi keputusan untuk menyetujui atau tidaknya sebuah peminjaman?
- Bagaimana cara membuat model Machine Learning dengan mengimplementasikan metode klasifikasi dalam memprediksikan biayakeputusan untuk menyetujui atau tidaknya sebuah peminjaman?

### Goals

Tujuan dari pembuatan proyek ini adalah sebagai berikut:
- Mendapatkan pemahaman mengenai faktor yang paling memengaruhi keputusan untuk menyetujui atau tidaknya sebuah peminjaman.
- Membuat beberapa model Machine Learning yang dapat mengimplementasikan kasus klasifikasi dalam memprediksikan keputusan untuk menyetujui atau tidaknya sebuah peminjaman.

## Analytic Approach

### Solution statements
Berikut merupakan tahapan penting yang dilakukan dalam proyek ini:
- Menghapus kolom yang tidak perlu.
- Menghapus missing value.
- Menghapus fitur collinear.
- Mengubah tipe data dari beberapa kolom.
- Melakukan encoding dengan Label Encoder.
- Membagi data train dan test dengan porsi 80% untuk training dan 20% untuk testing.
- Melakukan standardisasi data terhadap data numerik.
- Menggunakan data sampling dengan SMOTE.
- Melakukan modeling dengan algoritma Logistic Regression, K-Nearest Neighbor, Decision Tree, Naive Bayes, dan Random Forest.
- Dilakukan evaluasi model dengan membandingkan akurasi model.

## Data Understanding
Dataset yang digunakan pada proyek ini dapat di-download [disini](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv). Dataset berisikan 74 kolom yang memiliki 466285 sampel data. Detail dari masing-masing kolom dapat diuraikan sebagai berikut:
Atribut  | Keterangan
------------- | -------------
addr_state	|	Negara yang disediakan oleh peminjam dalam aplikasi pinjaman
annual_inc	|	Penghasilan tahunan yang dilaporkan sendiri yang disediakan oleh peminjam selama pendaftaran.
annual_inc_joint	|	Penghasilan tahunan yang dilaporkan sendiri gabungan yang disediakan oleh co-peminjam selama pendaftaran
application_type	|	Menunjukkan apakah pinjaman adalah aplikasi individu atau aplikasi bersama dengan dua peminjam bersama
collection_recovery_fee	|	Biaya pengumpulan biaya penagihan
collections_12_mths_ex_med	|	Jumlah koleksi dalam 12 bulan tidak termasuk koleksi medis
delinq_2yrs	|	Jumlah 30+ hari insiden kenakalan yang lewat dalam file kredit peminjam selama 2 tahun terakhir
desc	|	Deskripsi pinjaman yang disediakan oleh peminjam
dti	|	Rasio yang dihitung menggunakan total pembayaran utang bulanan peminjam atas total kewajiban utang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi dengan pendapatan bulanan peminjam yang dilaporkan sendiri.
dti_joint	|	Rasio yang dihitung menggunakan total pembayaran bulanan peminjam bersama atas total kewajiban utang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi oleh pendapatan bulanan yang dilaporkan sendiri oleh co-peminjam yang dilaporkan sendiri
earliest_cr_line	|	Bulan jalur kredit yang paling awal dilaporkan peminjam dibuka
emp_length	|	Panjang pekerjaan dalam beberapa tahun. Nilai yang mungkin adalah antara 0 dan 10 di mana 0 berarti kurang dari satu tahun dan 10 berarti sepuluh tahun atau lebih.
emp_title	|	Judul pekerjaan yang disediakan oleh peminjam saat mengajukan pinjaman.*
fico_range_high	|	Kisaran batas atas fico peminjam pada awal pinjaman.
fico_range_low	|	Kisaran batas bawah fico peminjam pada awal pinjaman.
funded_amnt	|	Jumlah total yang berkomitmen untuk pinjaman itu pada saat itu.
funded_amnt_inv	|	Jumlah total yang dilakukan oleh investor untuk pinjaman itu pada saat itu.
grade	|	LC menugaskan nilai pinjaman
home_ownership	|	Status kepemilikan rumah yang disediakan oleh peminjam selama pendaftaran. Nilai -nilai kami adalah: sewa, sendiri, hipotek, lainnya.
id	|	ID yang ditugaskan LC yang unik untuk daftar pinjaman.
initial_list_status	|	Status daftar awal pinjaman. Nilai yang mungkin adalah - w, f
inq_last_6mths	|	Jumlah pertanyaan dalam 6 bulan terakhir (tidak termasuk pertanyaan otomatis dan hipotek)
installment	|	Pembayaran bulanan yang terutang oleh peminjam jika pinjaman berasal.
int_rate	|	Suku bunga pinjaman
is_inc_v	|	Menunjukkan jika pendapatan diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi
issue_d	|	Bulan yang didanai pinjaman
last_credit_pull_d	|	Bulan terbaru LC menarik kredit untuk pinjaman ini
last_fico_range_high	|	Rentang batas atas yang ditarik oleh fico terakhir peminjam.
last_fico_range_low	|	Rentang batas bawah yang ditarik oleh fico terakhir peminjam.
last_pymnt_amnt	|	Jumlah total pembayaran terakhir yang diterima
last_pymnt_d	|	Bulan lalu pembayaran diterima
loan_amnt	|	Jumlah pinjaman yang terdaftar yang diterapkan oleh peminjam. Jika pada suatu titik waktu, departemen kredit mengurangi jumlah pinjaman, maka itu akan tercermin dalam nilai ini.
loan_status	|	Status pinjaman saat ini
member_id	|	ID yang ditugaskan LC yang unik untuk anggota peminjam.
mths_since_last_delinq	|	Jumlah bulan sejak kenakalan terakhir peminjam.
mths_since_last_major_derog	|	Bulan sejak peringkat 90 hari atau lebih buruk terakhir
mths_since_last_record	|	Jumlah bulan sejak catatan publik terakhir.
next_pymnt_d	|	Tanggal Pembayaran Terjadwal Berikutnya
open_acc	|	Jumlah jalur kredit terbuka dalam file kredit peminjam.
out_prncp	|	Kepala sekolah yang tersisa untuk jumlah total yang didanai
out_prncp_inv	|	Kepala sekolah yang tersisa untuk sebagian dari jumlah total yang didanai oleh investor
policy_code	|	Policy_code yang tersedia untuk umum = 1
pub_rec	|	Jumlah catatan publik yang menghina
purpose	|	Kategori yang disediakan oleh peminjam untuk permintaan pinjaman.
pymnt_plan	|	Menunjukkan jika rencana pembayaran telah diberlakukan untuk pinjaman
recoveries	|	Posting biaya pemulihan kotor
revol_bal	|	Total Saldo Revolving Credit
revol_util	|	Tingkat pemanfaatan jalur bergulir, atau jumlah kredit yang digunakan peminjam relatif terhadap semua kredit revolving yang tersedia.
sub_grade	|	LC Ditugaskan Subgrade Pinjaman
term	|	Jumlah pembayaran pinjaman. Nilai dalam beberapa bulan dan dapat berupa 36 atau 60.
title	|	Judul pinjaman yang disediakan oleh peminjam
total_acc	|	Jumlah total jalur kredit saat ini dalam file kredit peminjam
total_pymnt	|	Pembayaran diterima hingga saat ini untuk jumlah total yang didanai
total_pymnt_inv	|	Pembayaran diterima hingga saat ini untuk sebagian dari jumlah total yang didanai oleh investor
total_rec_int	|	Bunga diterima hingga saat ini
total_rec_late_fee	|	Biaya keterlambatan yang diterima hingga saat ini
total_rec_prncp	|	Kepala sekolah diterima hingga saat ini
url	|	URL untuk halaman LC dengan data daftar.
verified_status_joint	|	Menunjukkan jika pendapatan bersama co-peminjam diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi
zip_code	|	3 nomor pertama dari kode pos yang disediakan oleh peminjam dalam aplikasi pinjaman.
open_acc_6m	|	Jumlah perdagangan terbuka dalam 6 bulan terakhir
open_il_6m	|	Jumlah perdagangan angsuran aktif saat ini
open_il_12m	|	Jumlah akun angsuran yang dibuka dalam 12 bulan terakhir
open_il_24m	|	Jumlah akun angsuran yang dibuka dalam 24 bulan terakhir
mths_since_rcnt_il	|	Bulan sejak akun angsuran terbaru dibuka
total_bal_il	|	Total saldo saat ini dari semua akun angsuran
il_util	|	Rasio total saldo saat ini dengan batas kredit/kredit tinggi pada semua instal acct
open_rv_12m	|	Jumlah perdagangan revolving dibuka dalam 12 bulan terakhir
open_rv_24m	|	Jumlah perdagangan revolving dibuka dalam 24 bulan terakhir
max_bal_bc	|	Saldo arus maksimum terutang pada semua akun bergulir
all_util	|	Saldo ke batas kredit untuk semua perdagangan
total_rev_hi_lim	|	Total Batas Kredit/Kredit Tinggi Revolving
inq_fi	|	Jumlah pertanyaan keuangan pribadi
total_cu_tl	|	Jumlah Perdagangan Keuangan
inq_last_12m	|	Jumlah pertanyaan kredit dalam 12 bulan terakhir
acc_now_delinq	|	Jumlah akun di mana peminjam sekarang nakal.
tot_coll_amt	|	Total jumlah pengumpulan yang pernah ada
tot_cur_bal	|	Total Saldo Saat Ini dari Semua Akun

Teknik EDA yang dilakukan:
Gambar  | Keterangan
------------- | -------------
![corr](https://i.ibb.co/hyJWL37/correlation.png) | Dilakukan visualisasi dengan heatmap untuk melihat korelasi antar kolom numerikal untuk melihat hubungan variabel manakah yang paling dominan. Dengan visualisasi ini juga dapat ditemukan fitur collinear pada dataset.

## Data Preparation
Pada tahap ini dilakukan preprocessing untuk menghasilkan data yang sudah siap pakai untuk dimodelkan, tahapan yang dilalui diuraikan sebagai berikut:
- Pertama dilakukan pengecekan data, mulai dari menghapus kolom yang tidak perlu, menghapus missing value, menghapus fitur collinear, dan mengubah tipe data yang tidak sesuai sehingga informasinya dapat berguna untuk model.
- Kedua, dilakukan transformasi terhadap kolom kategorikal (biasa disebut **encoding**) menggunakan Label Encoder karena data kategorikal yang ada cenderung bersifat ordinal data.
- Ketiga, dilakukan pembagian data menjadi data latih dan data uji di mana 80% digunakan sebagai data latih, dan 20% sisanya sebagai data uji.
- Keempat, dilakukan transformasi terhadap kolom numerikal dengan proses standardisasi untuk menyamakan skala data sehingga algoritma Machine Learning dapat memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan.
- Kelima, dilakukan data sampling dengan SMOTE untuk mengatasi inbalanced data.

## Modeling
Setelah melakukan tahapan preprocessing maka data telah siap digunakan untuk pemodelan. Pada proyek ini dilakukan pembuatan model menggunakan 5 (empat) algoritma yang berbeda yaitu Logistic Regression, K-Nearest Neighbor, Decision Tree, Naive Bayes, dan Random Forest.

## Evaluation
Berikut adalah hasil akurasi dari tiap model dengan dan tanpa data sampling.
Tanpa SMOTE  | Dengan SMOTE
------------- | -------------
![corr](https://i.ibb.co/jV84FYs/tanpa-smote.png) | ![corr](https://i.ibb.co/bN3Hvt9/dengan-smote.png)

Seperti yang dapat dilihat dari hasil di atas, baik dengan atau tanpa data sampling, model berbasis Tree berkinerja jauh lebih baik dibanding model yang lain. Memang, algoritma berbasis Tree menawarkan stabilitas dan kemampuan interpretasi yang tinggi untuk prediksi pemodelan. Algoritma tersebut memetakan interaksi nonlinier dengan cukup baik, tidak seperti model linier. Algoritma tersebut dapat beradaptasi dengan situasi apa pun dan memecahkan tantangan apa pun (klasifikasi atau regresi).