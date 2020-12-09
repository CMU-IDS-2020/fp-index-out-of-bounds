# CMU Interactive Data Science Final Project

* **Online URL**: Too large to deploy, please see instrunction below for local deployment
* **Team members**:
  * Contact person: jingyua4@andrew.cmu.edu
  * chuhanf@andrew.cmu.edu
  * jiechenx@andrew.cmu.edu
  * TODO@andrew.cmu.edu
* **Track**: Model

## Work distribution

Update towards the end of the project.

## Deliverables

### Proposal

- [X] The URL at the top of this readme needs to point to your application online. It should also list the names of the team members.
- [X] A completed proposal. The contact should submit it as a PDF on Canvas.

### Design review

- [X] Develop a prototype of your project.
- [X] Create a 5 minute video to demonstrate your project and lists any question you have for the course staff. The contact should submit the video on Canvas.

### Final deliverables

- [X] All code for the project should be in the repo.
- [X] A 5 minute video demonstration.
- [X] Update Readme according to Canvas instructions.
- [ ] A detailed project report. The contact should submit the video and report as a PDF on Canvas.

### Deployment Instruction
- Please clone the entire github repo to your local repository, all the dependencies are recorded in the **requirement.txt** (NOT gector.requirement.txt).
- Please make sure you are using python 3.7 (python 3.8 will likely raise errors on the code) and correctly installed all the required dependencies using
`pip3 install -r requirement.txt`
- In order for the model to load, please download the check point directly using [this link](https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gector.th) and place the downloaded checkpoint file in the **main project directory**.
- The downloading chould take a while, since the model check point is over 500 Mb.
- Nevigate to the project directory, run `streamlit run streamlit_app.py` to open the steamlit application. 
*Please Note:* The first time running the application would take a while since the models will need to download nlp corpus and package spicific data. The downloading process can be tracked using consoles.


