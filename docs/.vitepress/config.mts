import { defineConfig } from 'vitepress'

const enNav = [
  {
    text: 'Essentials',
    items: [
      { text: 'Preface', link: '/en/前言' },
      { text: 'Tooling', link: '/en/chapter1/wandb使用介绍' },
      { text: 'Tokenizer', link: '/en/chapter2/chapter2_分词器' },
      { text: 'PyTorch and Resource Accounting', link: '/en/chapter3/chapter3_pytorch与资源核算' },
      { text: 'Architecture and Training Details', link: '/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: 'MOE', link: '/en/chapter5/chapter5_混合专家模型' },
      { text: 'GPU Optimization', link: '/en/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: 'Data Engineering', link: '/en/chapter11/chapter11_数据工程' },
      { text: 'Training Pipeline', link: '/en/chapter13/chapter13_第十三章大模型的基本训练流程' },
      { text: 'Evaluation and Benchmarks', link: '/en/chapter12/chapter12_评估与基准测试' },
    ],
  },
  {
    text: 'Advanced',
    items: [
      { text: 'Distributed Training', link: '/en/chapter8/chapter8_第八章分布式训练' },
      { text: 'GPU High Performance Programming', link: '/en/chapter7/chapter7_第七章GPU高性能编程' },
      { text: 'Scaling Laws', link: '/en/chapter9/chapter9_Scaling_Laws' },
      { text: 'Inference', link: '/en/chapter10/推理' },
      { text: 'RLVR', link: '/en/chapter14/chapter14_可验证奖励的强化学习' },
    ],
  },
  {
    text: 'Extra Chapters',
    items: [{ text: 'Chapter 15', link: '/en/chapter15/什么是LLM推理' },{ text: 'Chapter 15', link: '/en/chapter15/LLM的未来LeCun' }]
  },
]

const enSidebar = {
  '/en/': [
    {
      text: 'Chapters',
      items: [
        { text: 'Preface', link: '/en/前言' },
        { text: 'Chapter 1 Tooling', link: '/en/chapter1/wandb使用介绍' },
        { text: 'Chapter 2 Tokenizer', link: '/en/chapter2/chapter2_分词器' },
        { text: 'Chapter 3 PyTorch and Resource Accounting', link: '/en/chapter3/chapter3_pytorch与资源核算' },
        { text: 'Chapter 4 Language Model Architecture and Training Details', link: '/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
        { text: 'Chapter 5 Mixture of Experts', link: '/en/chapter5/chapter5_混合专家模型' },
        { text: 'Chapter 6 GPU and Related Optimization', link: '/en/chapter6/chapter6_第六章GPU和GPU相关的优化' },
        { text: 'Chapter 7 GPU High Performance Programming', link: '/en/chapter7/chapter7_第七章GPU高性能编程' },
        { text: 'Chapter 8 Distributed Training', link: '/en/chapter8/chapter8_第八章分布式训练' },
        { text: 'Chapter 9 Scaling Laws', link: '/en/chapter9/chapter9_Scaling_Laws' },
        { text: 'Chapter 10 Inference', link: '/en/chapter10/推理' },
        { text: 'Chapter 11 Data Engineering', link: '/en/chapter11/chapter11_数据工程' },
        { text: 'Chapter 12 Evaluation and Benchmarks', link: '/en/chapter12/chapter12_评估与基准测试' },
        { text: 'Chapter 13 Basic Training Pipeline for LLMs', link: '/en/chapter13/chapter13_第十三章大模型的基本训练流程' },
        { text: 'Chapter 14 Reinforcement Learning with Verifiable Rewards', link: '/en/chapter14/chapter14_可验证奖励的强化学习' },
        { text: 'Chapter 15 Extended Content', items: [
          { text: '1. What is LLM reasoning？', link: '/en/chapter15/什么是LLM推理' },
          { text: '2. LLM Future - LeCun', link: '/en/chapter15/LLM的未来LeCun' },
        ]},
      ],
    },
  ],
}

const zhNav = [
  {
    text: '基础必学',
    items: [
      { text: '前言', link: '/前言' },
      { text: '工具使用', link: '/chapter1/wandb使用介绍' },
      { text: '分词器', link: '/chapter2/chapter2_分词器' },
      { text: 'PyTorch与资源核算', link: '/chapter3/chapter3_pytorch与资源核算' },
      { text: '架构与细节', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: 'MOE', link: '/chapter5/chapter5_混合专家模型' },
      { text: 'GPU优化', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: '数据工程', link: '/chapter11/chapter11_数据工程' },
      { text: '训练流程', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
      { text: '评估与基准测试', link: '/chapter12/chapter12_评估与基准测试' },
    ],
  },
  {
    text: '进阶选修',
    items: [
      { text: '分布式训练', link: '/chapter8/chapter8_第八章分布式训练' },
      { text: 'GPU高性能编程', link: '/chapter7/chapter7_第七章GPU高性能编程' },
      { text: 'Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
      { text: '推理', link: '/chapter10/推理' },
      { text: 'RLVR', link: '/chapter14/chapter14_可验证奖励的强化学习' },
    ],
  },
  {
    text: '扩展内容',
    items: [{ text: '第15章-1.什么是LLM推理', link: '/chapter15/什么是LLM推理' },{ text: '第15章-2.LLM的未来LeCun', link: '/chapter15/LLM的未来LeCun' }],
  },
]

const zhSidebar = {
  '/': [
    {
      text: '章节',
      items: [
        { text: '前言', link: '/前言' },
        { text: '第1章 工具使用', link: '/chapter1/wandb使用介绍' },
        { text: '第2章 分词器', link: '/chapter2/chapter2_分词器' },
        { text: '第3章 PyTorch 与资源核算', link: '/chapter3/chapter3_pytorch与资源核算' },
        { text: '第4章 语言模型架构与训练细节', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
        { text: '第5章 混合专家模型', link: '/chapter5/chapter5_混合专家模型' },
        { text: '第6章 GPU 与相关优化', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
        { text: '第7章 GPU 高性能编程', link: '/chapter7/chapter7_第七章GPU高性能编程' },
        { text: '第8章 分布式训练', link: '/chapter8/chapter8_第八章分布式训练' },
        { text: '第9章 Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
        { text: '第10章 推理', link: '/chapter10/推理' },
        { text: '第11章 数据工程', link: '/chapter11/chapter11_数据工程' },
        { text: '第12章 评估与基准测试', link: '/chapter12/chapter12_评估与基准测试' },
        { text: '第13章 大模型的基本训练流程', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
        { text: '第14章 可验证奖励的强化学习', link: '/chapter14/chapter14_可验证奖励的强化学习' },
        { text: '第15章 扩展内容', items: [
          { text: '1. 什么是LLM推理？', link: '/chapter15/什么是LLM推理' },
          { text: '2. LLM 的未来 - Lecun', link: '/chapter15/LLM的未来LeCun' },
        ]},
      ],
    },
  ],
}

export default defineConfig({
  title: 'Diy-LLM',
  description: '面向中文学习者的大语言模型系统化学习课程。',
  base: '/diy-llm/',
  ignoreDeadLinks: true,
  rewrites: {
    'zh/:rest*': ':rest*',
  },
  head: [
    ['link', { rel: 'icon', href: '/diy-llm/datawhale.png' }],
    ['meta', { name: 'theme-color', content: '#2563eb' }],
  ],
  markdown: {
    math: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },
  themeConfig: {
    logo: '/datawhale.png',
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/diy-llm' },
    ],
    outline: {
      level: [2, 3],
    },
  },
  locales: {
    root: {
      label: '中文',
      lang: 'zh-CN',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outlineTitle: '本页目录',
        returnToTopLabel: '返回顶部',
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
      },
    },
    en: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      srcDir: 'en',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        outlineTitle: 'On this page',
        returnToTopLabel: 'Back to top',
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        docFooter: {
          prev: 'Previous page',
          next: 'Next page',
        },
      },
    },
  },
})
