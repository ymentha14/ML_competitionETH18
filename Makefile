.PHONY: all clean clear test archive

initialize:
	module load python
	module load python/3.7.1
# starts the run for all json present in the input file
submit:
	@bsub make all_run
triple:
	@ls $$PWD/input_params/cross_three/* | xargs -I {} json validate --schema-file={}
	@python3 triple_main.py

all_run: 
	#make all_run > lsf0001.txt 2>&1
	@ls $$PWD/input_params/regular_input/* | xargs -I {} json validate --schema-file={}
	@ls $$PWD/input_params/regular_input/* | xargs -I {} python3 main.py {}
	@mpg123 sound/wmelon.mp3

all_run_marijn:
	@ls $$PWD/input_params/regular_input/* | xargs -I {} json validate --schema-file={}
	@ls $$PWD/input_params/regular_input/* | xargs -I {} python3 main_marijn.py {}

FOLDER := $(shell ls archive | sort -nr | head -n 1)

#clear the log directory
clear_log:
	@rm -r logs

#clear the result directory
clear_results:
	@rm -r results

#display the best current solution in all folders (logs and archive)
sort:
	@grep -nr "accuracy\":" archive/$(FOLDER)/ | awk '{print $$NF,$$0}'  | sort -rn | cut -f2- -d' '
	

#copy everything in the logs file into the archive file
archive:
	./prog_archive.sh

#move everything from the log to the most recent archive
move:
	@cp -r logs/* archive/$(FOLDER)/

#run the script allowing to asssess the pca parameters
pca:
	@python3 analyze_pca.py

#displays the results of the logs in tensorboard
display:
	@tensorboard --logdir logs&
	@sleep 4
	#@xdg-open http://localhost:6006/
	#open your browser, type "localhost:6006"

# runs the file with the arguments present in the code
run: main.py
	#@python3 main.py
	@python3.7 main.py
	
#reset the tensorboard
reset:
	@kill $$(ps aux | grep tensorboard | awk '{print $$2}')

#display the archives. Can be crowded.
display_arch:
	@tensorboard --logdir archives&
	@sleep 4
	@xdg-open http://localhost:6006/


#bsub -W 1:30 -R "rusage[mem=10000]"
