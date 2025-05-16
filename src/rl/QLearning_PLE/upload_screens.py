from huggingface_hub import upload_large_folder, HfApi



def upload_screens():
    upload_large_folder(
        folder_path="screens",  # path to your local folder
        repo_id="DiffusionArcade/Pong",
        repo_type="dataset",              # specify this is a dataset repo
        allow_patterns=["images/damian_screens_03/**.jpg"]
        # allow_patterns=["*.jpg", "*.png"], # optional: only include image files
        # ignore_patterns=["screens/*.jpg"]
        # ignore_patterns=None,             # optional: skip files
    )



if __name__ == "__main__":
    upload_screens()
    print("Upload complete.")