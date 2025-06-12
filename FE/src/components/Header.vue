<script setup>
import { onMounted } from 'vue';
import { RouterLink } from 'vue-router'

const props = defineProps({
  user: Object
})

onMounted(() => {
  console.log('user', props.user)
})
</script>

<template>
  <header class="relative flex flex-wrap sm:justify-start sm:flex-nowrap w-full text-sm py-3">
    <nav class="max-w-[85rem] w-full mx-auto sm:flex sm:items-center sm:justify-between">
      <div class="flex items-center justify-between px-4">
        <RouterLink class="flex-none text-xl font-semibold focus:outline-hidden focus:opacity-80" to="/"
          aria-label="Brand">
          Stress Cope
        </RouterLink>
        <div class="sm:hidden">
          <button type="button"
            class="hs-collapse-toggle relative size-9 flex justify-center items-center gap-x-2 rounded-lg border border-gray-200 bg-white text-gray-800 shadow-2xs hover:bg-gray-50 focus:outline-hidden focus:bg-gray-50 disabled:opacity-50 disabled:pointer-events-none"
            id="navigation-collapse" aria-expanded="false" aria-controls="navigation"
            aria-label="Toggle navigation" data-hs-collapse="#navigation">
            <svg class="hs-collapse-open:hidden shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24"
              height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
              stroke-linejoin="round">
              <line x1="3" x2="21" y1="6" y2="6" />
              <line x1="3" x2="21" y1="12" y2="12" />
              <line x1="3" x2="21" y1="18" y2="18" />
            </svg>
            <svg class="hs-collapse-open:block hidden shrink-0 size-4" xmlns="http://www.w3.org/2000/svg" width="24"
              height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
              stroke-linejoin="round">
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
            <span class="sr-only">Toggle navigation</span>
          </button>
        </div>
      </div>
      <div id="navigation"
        class="hidden hs-collapse overflow-hidden transition-all duration-300 basis-full grow sm:block m-4"
        aria-labelledby="navigation-collapse">
        <div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-end sm:mt-0 sm:ps-5">
          <RouterLink to="/" class="font-medium text-white py-2 px-4 rounded-lg hover:bg-cyan-700 transition duration-300"
            activeClass="bg-cyan-600">Beranda</RouterLink>
          <RouterLink to="/cek-stres" class="font-medium text-white py-2 px-4 rounded-lg hover:bg-cyan-700 transition duration-300"
            activeClass="bg-cyan-600">Cek Stres</RouterLink>
          <RouterLink to="/riwayat" class="font-medium text-white py-2 px-4 rounded-lg hover:bg-cyan-700 transition duration-300">Riwayat</RouterLink>
        </div>
      </div>
    </nav>
  </header>
</template>