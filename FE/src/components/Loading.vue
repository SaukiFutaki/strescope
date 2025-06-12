<script setup>
const props = defineProps({
  size: Number,
  default: 4
})
</script>

<template>
  <div class="animate-spin inline-block border-3 border-current border-t-transparent text-blue-600 rounded-full"
    :class="`size-${props.size}`" role="status" aria-label="loading">
    <span class="sr-only">Loading...</span>
  </div>
</template>