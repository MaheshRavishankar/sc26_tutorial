#!/usr/bin/env bash
# Replay the solution branches on top of main.
#
# Each solution branch is exactly one commit -- the answers to one part's
# exercises -- sitting on a base branch:
#
#   main
#    +- session1-part1-solution          session 1 part 1 solved
#    |   +- session1-part2-solution      ... plus session 1 part 2 solved
#    +- session2-part1-solution          session 2 part 1 solved
#    +- session2-part2-solution          session 2 part 2 solved
#
# Run this after main moves, then force-push whatever changed:
#
#   scripts/refresh_solution_branches.sh
#   git push --force-with-lease origin session1-part1-solution ...
#
# A conflict means main changed a file a solution also changes: resolve it,
# `git cherry-pick --continue`, then re-run this script.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

start_branch=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

# This repo deliberately has no user.name/user.email configured, so the
# cherry-picks below would have no committer. Borrow main's last author.
if ! git config user.email >/dev/null; then
  GIT_COMMITTER_NAME=$(git log -1 --format='%an' main)
  GIT_COMMITTER_EMAIL=$(git log -1 --format='%ae' main)
  GIT_AUTHOR_NAME=$GIT_COMMITTER_NAME
  GIT_AUTHOR_EMAIL=$GIT_COMMITTER_EMAIL
  export GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree is dirty -- commit or stash first" >&2
  exit 1
fi

replay() { # <base> <branch>
  local base=$1 branch=$2 tip
  tip=$(git rev-parse "$branch")
  if git merge-base --is-ancestor "$base" "$branch" &&
     [ "$(git rev-list --count "$base..$branch")" = "1" ]; then
    echo "==> $branch already sits on $base, nothing to do"
    return
  fi
  echo "==> replaying $branch onto $base"
  git switch --quiet --detach "$base"
  git cherry-pick "$tip"
  git branch --force "$branch" HEAD
}

replay main                   session1-part1-solution
replay session1-part1-solution session1-part2-solution
replay main                   session2-part1-solution
replay main                   session2-part2-solution

git switch --quiet "$start_branch"
echo
git --no-pager log --oneline --graph --decorate \
  main session1-part1-solution session1-part2-solution \
  session2-part1-solution session2-part2-solution | head -20
