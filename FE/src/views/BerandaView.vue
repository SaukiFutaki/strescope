<script setup>
import gambar from '@/assets/menikmati-hidup.svg'
import { useRouter } from 'vue-router';

const router = useRouter();

const getStarted = (sudahMengetahuiBPM) => {
  const modal = new HSOverlay(document.querySelector('#get-started-modal'));

  modal.close();

  if (sudahMengetahuiBPM) {
    router.push({ name: 'cek-stres' })
  } else {
    router.push({ name: 'hitung-bpm' })
  }
}
</script>

<template>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-10 items-center my-17">
    <img :src="gambar" alt="Menikmati Hidup" class="w-2/3 mx-auto">

    <div>
      <h1 class="text-2xl font-bold mb-3">
        Cek tingkat stresmu dengan mudah dan cepat
      </h1>

      <p class="text-lg mb-5">
        Prediksi tingkat stres harianmu berdasarkan data fisiologis & aktivitas menggunakan kecerdasan buatan.
      </p>

      <button type="button"
        class="py-2 px-4 inline-flex items-center gap-x-2 text-sm font-medium rounded-lg border border-transparent bg-cyan-600 text-white hover:bg-cyan-700 focus:outline-hidden focus:bg-cyan-700 disabled:opacity-50 disabled:pointer-events-none"
        aria-haspopup="dialog" aria-expanded="false" aria-controls="get-started-modal"
        data-hs-overlay="#get-started-modal">
        Get Started
      </button>

      <div id="get-started-modal"
        class="hs-overlay [--overlay-backdrop:static] hidden size-full fixed top-0 start-0 z-80 overflow-x-hidden overflow-y-auto pointer-events-none"
        role="dialog" tabindex="-1" aria-labelledby="get-started-modal-label" data-hs-overlay-keyboard="false">
        <div
          class="hs-overlay-open:mt-7 hs-overlay-open:opacity-100 hs-overlay-open:duration-500 mt-0 opacity-0 ease-out transition-all sm:max-w-lg sm:w-full m-3 sm:mx-auto h-[calc(100%-56px)] min-h-[calc(100%-56px)] flex items-center">
          <div class="flex flex-col bg-white border border-gray-200 shadow-2xs rounded-xl pointer-events-auto">
            <div class="flex justify-between items-center py-3 px-4 border-b border-gray-200">
              <h3 id="get-started-modal-label" class="font-bold text-gray-800">
                Get Started
              </h3>
              <button type="button"
                class="size-8 inline-flex justify-center items-center gap-x-2 rounded-full border border-transparent bg-gray-100 text-gray-800 hover:bg-gray-200 focus:outline-hidden focus:bg-gray-200 disabled:opacity-50 disabled:pointer-events-none"
                aria-label="Close" data-hs-overlay="#get-started-modal">
                <span class="sr-only">Close</span>
                <svg class="shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24" height="24"
                  viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                  stroke-linejoin="round">
                  <path d="M18 6 6 18"></path>
                  <path d="m6 6 12 12"></path>
                </svg>
              </button>
            </div>
            <div class="p-4 overflow-y-auto">
              <p class="mt-1 text-gray-800">
                Apakah kamu sudah mengetahui berapa detak jantungmu?
              </p>
            </div>
            <div class="flex justify-end items-center gap-x-2 py-3 px-4 border-t border-gray-200">
              <button type="button" @click="getStarted(false)"
                class="py-2 px-3 inline-flex items-center gap-x-2 text-sm font-medium rounded-lg border border-gray-200 bg-white text-gray-800 shadow-2xs hover:bg-gray-50 focus:outline-hidden focus:bg-gray-50 disabled:opacity-50 disabled:pointer-events-none">
                Belum
              </button>
              <button type="button" @click="getStarted(true)"
                class="py-2 px-3 inline-flex items-center gap-x-2 text-sm font-medium rounded-lg border border-transparent bg-blue-600 text-white hover:bg-blue-700 focus:outline-hidden focus:bg-blue-700 disabled:opacity-50 disabled:pointer-events-none">
                Sudah
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
