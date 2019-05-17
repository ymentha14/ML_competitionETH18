.PHONY: all clean clear test archive
# runs the file with the arguments present in the code
run: main.py
	@python3 main.py

# starts the run for all json present in the input file
all_run: 
	@ls $$PWD/input_params/* | xargs -I {} json validate --schema-file={}
	@ ls $$PWD/input_params/* | xargs -I {} python3 main.py {}
	@mpg123 sound/wmelon.mp3

#copy everything in the logs file into the archive file
FOLDER := $(shell ls archive | sort -nr | head -n 1)

#display the best current solution in all folders (logs and archive)
sort:
	@grep -nr "\"accuracy\":" archive/$(FOLDER)/ | awk '{print $$NF,$$0}'  | sort -rn | cut -f2- -d' '

archive:
	./archive.sh

#run the script allowing to asssess the pca parameters
pca:
	@python3 analyze_pca.py
#displays the results of the logs in tensorboard
display:
	@tensorboard --logdir logs&
	@sleep 4
	@xdg-open http://localhost:6006/

#display the archives. Can be crowded.
display_arch:
	@tensorboard --logdir archives&
	@sleep 4
	@xdg-open http://localhost:6006/

#reset the tensorboard
reset:
	@kill $$(ps aux | grep tensorboard | awk '{print $$2}')

#clear the log directory
clear_log:
	@rm -r logs

#clear the result directory
clear_results:
	@rm -r results
