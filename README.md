# HACTrack

[PHALP' 모델 실행]
1. 4dhumans 깃허브의 "dev branch"를 git clone --branch dev https://github.com/shubham-goel/4D-Humans.git로 다운받기 (dev branch: PHALP', tex branch 이후 버전: 4DHumans)
2. 생성된 4D-Humans 폴더 내 Readme의 Installation and Setup의 안내를 따라 가상환경 생성(시간이 조금 소요되며 가상환경 이름은 4D-humans)
3. 생성한 4D-humans 가상환경에서 pip install git+https://github.com/brjathu/PHALP.git@dev로 dev 브랜치의 PHALP 라이브러리 설치
4. 4D-Humans 폴더에서 python track.py video.source="example_data/videos/gymnasts.mp4" 명령어로 PHALP' 실행 확인(필요 모델 및 파일 다운로드 될 것임)
5. pip install filterpy

[PoseTrack 데이터셋 test 코드 수정 및 데이터셋 다운로드]
1. 본 repository의 track.py로 track.py 대체
2. '_DATA' 파일을 ~를 통해 4D-Humans 폴더에 다운로드
3. PoseTrack validation 이미지, 어노테이션 데이터셋 다운로드하고 track.py line 69에 경로를 알맞게 수정
4. PHALP.py로 ~anaconda path/envs/4D-humans/lib/python3.10/site-packages/phalp/trackers/PHALP.py를 대체
5. PHALP.py line 195의 경로는 HACTrack 방법에 따라 재분류된 검출 결과 파일에 알맞게 수정
6. 본 repository의 tracker.py로 ~anaconda path/envs/4D-humans/lib/python3.10/site-packages/phalp/external/deep_sort/tracker.py 대체
7. 4D-Humans 폴더에서 python track.py 실행

[평가 코드 수정]
1. posetrack validation 데이터셋의 170개의 시퀀스에 대한 결과는 4D-Humans/outputs/results에 시퀀스 별로 pkl로 저장
2. https://github.com/JonathonLuiten/TrackEval.git으로 TrackEval 폴더를 다운로드 하고 python setup.py install로 라이브러리 설치한
3. 본 reposityroy의 create_txt.py로 eval.py를 대체하고 python eval.py ~4D-Humans경로/outputs/results phalp posetrack 명령어로 170개의 시퀀스를 posetrack_phalp.pkl로 통합 후 txt 폴더내 170개의 txt 파일로 변환
4. git clone https://github.com/anDoer/PoseTrack21.git으로 posetrack 평가 코드 다운로드
5. PoseTrack21/eval/posetrack21/scripts/run_mot.py
6. PoseTrack21/gt_processing.py으로 gt.txt process
