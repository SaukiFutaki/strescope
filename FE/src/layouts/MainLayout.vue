<script setup>
import Header from '@/components/Header.vue'
import Footer from '@/components/Footer.vue'
import User from '@/models/User'
import { useAuthStore } from '../stores/authStore';
import { onMounted, ref } from 'vue';

const authStore = useAuthStore()

const user = ref(new User({
  id: null,
  name: null,
  email: null
}))

onMounted(() => {
  // user.value = authStore.getUser
})
</script>

<template>
  <Header :user="user" />

  <main class="max-w-[85rem] w-full mx-auto p-5 pb-15">
    <slot />
  </main> 

  <Footer />
</template>
