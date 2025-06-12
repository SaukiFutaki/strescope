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
</script>

<template>
  <div class="riwayat">
    <h1>Ini halaman riwayat</h1>
  </div>
</template>

<style scoped>
@media (min-width: 1024px) {
  .riwayat {
    min-height: 100vh;
    display: flex;
    align-items: center;
  }
}
</style>
