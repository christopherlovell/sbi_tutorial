<script setup>
import { computed } from 'vue'
import katex from 'katex'

const props = defineProps({ expr: { type: String, required: true }, display: Boolean })
const html = computed(() =>
  katex.renderToString(props.expr, { displayMode: props.display, throwOnError: false }),
)
</script>

<template>
  <span class="tex" v-html="html" />
</template>
