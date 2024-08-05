context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
dir(context)
context.gitRepoUrl().get()