from huggingface_hub import upload_large_folder, HfApi



def upload_screens():
    upload_large_folder(
        folder_path="screens2",  # path to your local folder
        repo_id="DiffusionArcade/Pong_ep4999_allsteps_30fps",
        repo_type="dataset",              # specify this is a dataset repo
        allow_patterns=["images/damian_screens_02/**.jpg"]
        # allow_patterns=["*.jpg", "*.png"], # optional: only include image files
        # ignore_patterns=["screens/*.jpg"]
        # ignore_patterns=None,             # optional: skip files
    )



if __name__ == "__main__":
    upload_screens()
    print("Upload complete.")