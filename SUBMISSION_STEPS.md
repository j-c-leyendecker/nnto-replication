# Submitting to ReScience C: Daksh Anand

Everything is filled in and checked, including your GitHub user name (j-c-leyendecker).

* `nnto-replication/` is the repository to put on GitHub: code, results, tests, MIT
  `LICENSE`, `requirements.txt`, and the article in `article/` (`article.pdf`,
  `metadata.yaml`, LaTeX sources).
* Final checks (18 November 2026):
  * every number in the article was recomputed from the result files (56 checks);
  * all 28 references are cited and carry a DOI or URL;
  * the PDF builds without errors;
  * from a clean copy, with the original repository cloned next to it, the 107 tests pass and every figure regenerates.

## 1. Create the GitHub repository (5 min)
1. Make a free account at https://github.com if you do not have one.
2. Click **New repository**:
   * name `nnto-replication`;
   * **Public**;
   * do **not** add a README, licence or .gitignore, because the folder already has them.

## 2. Upload the folder
The folder has about 320 files and GitHub's web upload takes at most 100 at a time,
so use git (Terminal on your Mac) or GitHub Desktop.

```bash
cd ~/Downloads/ReScience_final/nnto-replication      # wherever you unzipped it
git init -b main
git add .
git commit -m "Replication package and article for ReScience C"
git remote add origin https://github.com/j-c-leyendecker/nnto-replication.git
git push -u origin main
```
GitHub asks for a personal access token instead of your password: Settings →
Developer settings → Personal access tokens. Alternatively, in GitHub Desktop, use
File → Add Local Repository → Publish repository, with "Keep this code private"
**unticked**.

Check that these open:
* https://github.com/j-c-leyendecker/nnto-replication (the README shows)
* https://github.com/j-c-leyendecker/nnto-replication/blob/main/article/article.pdf

## 3. Recommended: write to the original authors
Not required for a successful replication, but courteous, and editors like it. The
paper gives jnajmon@purdue.edu (Joel Najmon) and tovara@iupui.edu (Andres Tovar,
corresponding author). IUPUI has since been reorganized, so if that address bounces,
look up his current Purdue address. Draft:

> Subject: Replication of "Evaluation of Neural Network-Based Derivatives for Topology Optimization" (JMD 2024)
>
> Dear Dr. Najmon and Prof. Tovar,
>
> I am a student at Thapar Institute of Engineering and Technology (India). I have
> replicated your 2024 JMD article using the code, networks and data you published,
> which made the replication possible. I plan to submit it to ReScience C, a journal
> for replications, and wanted to share the main findings with you first:
>
> * Every number in the article that can be recomputed from the released material is
>   reproduced, and NNJ, CSM and AD agree to about 1e-15.
> * The CFD difference appears to come from single-precision rounding in the
>   optimization scripts: Model.predict rounds x ± 1e-6 to float32, so the step taken
>   is 1.0133e-6. With tf.keras.backend.set_floatx('float64'), CFD agrees to 2.3e-10.
> * In DBTO, repeated runs perturbed at the 1e-10 level end in two groups about 1.5 %
>   apart, so single runs may not rank the methods.
> * In HBTO, against a mesh-converged homogenization reference, the network's
>   derivative error dominates, and the final designs are 8.5–10.3 % less stiff
>   than predicted.
>
> The article and code are at https://github.com/j-c-leyendecker/nnto-replication.
> I would be grateful for any comments or corrections.
>
> Kind regards,
> Daksh Anand
> Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala

If they reply, the reply can be mentioned in the article during review.

## 4. Submit
Open https://github.com/ReScience/submissions/issues/new/choose and choose
**Replication**. Use the title
`[Re] Evaluation of Neural Network-Based Derivatives for Topology Optimization`
and paste:

```
**Original article:** Najmon, J. and Tovar, A. (2024). Evaluation of Neural Network-Based Derivatives for Topology Optimization. Journal of Mechanical Design 146(7), 071704. https://doi.org/10.1115/1.4064243

**PDF URL:** https://github.com/j-c-leyendecker/nnto-replication/blob/main/article/article.pdf
**Metadata URL:** https://github.com/j-c-leyendecker/nnto-replication/blob/main/article/metadata.yaml
**Code URL:** https://github.com/j-c-leyendecker/nnto-replication

**Scientific domain:** Computational Mechanics
**Programming language:** Python
**Suggested editor:**
```
Leave "Suggested editor" empty, or pick a name from https://rescience.github.io/board/.

## 5. During and after review
* An editor takes the issue and invites reviewers, and the review happens in the
  issue comments. Reply there, change the code or the article, and push again with
  `git add . && git commit -m "..." && git push`.
* After acceptance:
  * Archive the repository on https://zenodo.org: log in with GitHub, switch the
    repository on, and make a GitHub release. This gives the code a DOI.
  * Put that DOI and the editor's information into `article/metadata.yaml`.
  * The editor publishes the article.

## If you ever change the article
Edit `article/content.tex` or `article/metadata.yaml`, then rebuild the PDF on
Overleaf. Choose New Project → Upload Project and upload the `article/` folder as a
zip. Then set Menu → Compiler to **XeLaTeX** and the main document to `article.tex`.
If you edited `metadata.yaml`, also edit `metadata.tex` to match, because Overleaf
reads `metadata.tex`. Overleaf sets the references in the official ReScience style.
Save the new PDF as `article/article.pdf`, then commit and push.
