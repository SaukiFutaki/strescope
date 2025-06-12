<script setup>
import { onMounted, ref } from 'vue';

const isLoading = ref(true)

onMounted(() => {
  // TODO:
  // fetch api daftar riwayat
  // await fetch
  // update daftarPertanyaan dan isLoading
  // isLoading.value = false
})

const kategoriStress = (score) => {
  const kategori = [
    {
      minScore: 0,
      kategori: 'Stress Rendah',
      deskripsi: 'Selamat! Tingkat stres anda berada dalam tingkat yang sangat rendah. Tetap lakukan aktivitas yang membuat Anda nyaman dan bahagia!',
    },
    {
      minScore: 45,
      kategori: 'Stress Menengah',
      deskripsi: 'Anda menunjukkan tingkat stres menengah. Prioritaskan kesehatan mental dan fisik Anda. Lanjutkan kebiasaan baik dan lakukan hal yang anda senangi untuk meredakan ketegangan.',
    },
    {
      minScore: 85,
      kategori: 'Stress Tinggi',
      deskripsi: 'Tingkat stres Anda saat ini berada pada level tinggi. Segera mengambil langkah-langkah untuk mengelola dan menurunkannya. Pertimbangkan untuk mencari dukungan atau saran profesional',
    }
  ]

  if (score < kategori[0].minScore) return kategori[0]
  if (score < kategori[1].minScore) return kategori[1]
  return kategori[2]
}
</script>

<template>
  <div class="card">
    <div class="max-w-150 mx-auto">
      <h3 class="text-lg font-medium text-center">
        Stress Score
      </h3>
      <h1 class="text-3xl font-bold text-center">85%</h1>
      <p class="my-3 text-center">
        {{ kategoriStress(85).deskripsi }}
      </p>
    </div>
  </div>
</template>
