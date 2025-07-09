# KTds_MSAI_2
project risk inspector AI Agent

LINK : https://user32-webapp-euahh4h5e4exbehn.swedencentral-01.azurewebsites.net


![risk_inspector_home](https://github.com/user-attachments/assets/9da47d08-e7f2-45c4-989b-b45f75abbef1)

📘 Azure 기반 생성형 AI 프로젝트 제안서<br><br>
________________________________________
✅ 프로젝트: SI 프로젝트 리스크 탐지 및 대응 지원 에이전트<br>
📌 개요 및 목적<br>
SI 프로젝트의 회의록, 이슈 로그, 일정 변경 이력 등의 데이터를 GPT 기반 AI가 분석하여 잠재적인 일정 지연, 요구사항 변경, 인력 리스크 등 다양한 프로젝트 리스크를 사전에 탐지하고 우선순위를 매긴 후 대응 방안을 제안하는 리스크 인텔리전스 에이전트입니다<br>
  •	프로젝트 실패 요소 사전 감지 및 경고<br>
  •	프로젝트 리스크 가시화 및 실시간 모니터링<br>
  •	대응 조치 추천을 통한 PM 의사결정 지원<br><br>

🔧 활용 기술 및 Azure 서비스<br>
  •	Azure OpenAI (GPT-4.1 mini): 회의록/이슈 내용을 분석하여 리스크 요약 및 대응 가이드 생성<br>
  •	Azure Cognitive Search or Azure Storage Blob service : 회의록/이슈 로그 데이터 색인 및 조회<br>
  •	Streamlit + Azure App Service: 사용자 친화적인 리스크 분석 대시보드 제공<br><br>
  
🧩 아키텍처<br>
🎯 기대 효과<br>
  •	리스크 탐지 → 대응 보고서 생성 시간 1시간 → 5분 이내 단축<br>
  •	잠재 일정 지연, 변경 리스크 사전 탐지율 30%~40% 향상<br>
  •	프로젝트 운영 리스크 가시화로 PM의 대응력 및 의사결정 효율 향상<br>
  •	회의록 등 텍스트 중심 자료의 활용도 극대화<br><br>
  
⚠️ 구현 시 고려사항<br>
  •	회의록/이슈로그 표준화 필요 (CSV, JSON 등 일정 구조 유지)<br>
  •	비정형 데이터 분석의 한계로 인한 오탐/누락 가능성 → 지속적 프롬프트 개선 필요<br>
  •	GPT 응답의 근거 명시 및 원문 링크 제공 필요
