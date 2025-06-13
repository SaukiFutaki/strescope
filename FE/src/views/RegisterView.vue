<script setup>
import { ref } from 'vue';
import { RouterLink } from 'vue-router';

const isLoading = ref(false)

const register = () => {
  // TODO:
  // isLoading.value = true
  // kirim request post api register
  // await fetch
  // isLoading.value = false
}
</script>

<template>
  <div class="card max-w-120 mx-auto my-17">
    <h2 class="text-xl font-medium text-center mb-5">Silahkan lengkapi formulir berikut</h2>
    <div class="mt-5">
      <form id="registerForm" autocomplete="off">
        <div id="message" class="mb-5 border rounded-lg py-2 px-4" role="alert" style="display: none"></div>

        <div class="grid gap-y-4">
          <!-- Form Group -->
          <div class="mb-3">
            <label for="nameInput" class="block text-sm mb-2">Nama Lengkap</label>
            <div class="relative">
              <input type="text" id="nameInput" name="name"
                class="py-2 px-3 block w-full border-gray-200 rounded-sm text-sm focus:border-cyan-700 focus:ring-cyan-700"
                required="" />
            </div>
          </div>
          <!-- End Form Group -->

          <!-- Form Group -->
          <div class="mb-3">
            <label for="email" class="block text-sm mb-2">Email</label>
            <div class="relative">
              <input type="email" id="email" name="email"
                class="py-2 px-3 block w-full border-gray-200 rounded-sm text-sm focus:border-cyan-700 focus:ring-cyan-700"
                required="" />
            </div>
          </div>
          <!-- End Form Group -->

          <!-- Form Group -->
          <div class="mb-3">
            <label for="password" class="block text-sm mb-2">Password</label>
            <div class="relative">
              <input id="password" type="password" name="password"
                class="py-2.5 sm:py-3 ps-4 pe-10 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
                required />
              <button type="button" data-hs-toggle-password='{
                  "target": "#password"
                }'
                class="absolute inset-y-0 end-0 flex items-center z-20 px-3 cursor-pointer text-gray-400 rounded-e-md focus:outline-hidden focus:text-blue-600">
                <svg class="shrink-0 size-3.5" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path class="hs-password-active:hidden" d="M9.88 9.88a3 3 0 1 0 4.24 4.24"></path>
                  <path class="hs-password-active:hidden"
                    d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"></path>
                  <path class="hs-password-active:hidden"
                    d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"></path>
                  <line class="hs-password-active:hidden" x1="2" x2="22" y1="2" y2="22"></line>
                  <path class="hidden hs-password-active:block" d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z">
                  </path>
                  <circle class="hidden hs-password-active:block" cx="12" cy="12" r="3"></circle>
                </svg>
              </button>
            </div>
          </div>
          <!-- End Form Group -->

          <button type="button" @click="register"
            class="py-2 px-3 inline-flex justify-center items-center gap-2 rounded-sm border border-transparent font-semibold bg-cyan-700 text-white hover:bg-cyan-700 focus:outline-none focus:ring-0 focus:ring-cyan-700 focus:ring-offset-0 transition-all text-sm">
            Sign up
          </button>
        </div>
      </form>
      <!-- End Form -->

      <div class="mt-5 text-center">
        <p class="text-sm text-gray-500">
          Sudah punya akun?
          <RouterLink to="/login" class="font-semibold text-gray-700 hover:text-gray-800">
            Masuk
          </RouterLink>
        </p>
      </div>
    </div>
  </div>
</template>