from bsuite.sweep import SWEEP
from examples.impala import util
from redis import Redis

import run


class Args(run.Args):
    rank: int  # integer identifying the rank of this process


def main(args: Args):
    redis = Redis(host="redis")
    logger = util.AbslLogger()
    done = False
    while not done:
        id_or_done = redis.lpop("env-queue").decode("utf-8")
        if id_or_done == "DONE":
            logger.write(f"Process {args.rank} is done.")
            return
        assert id_or_done in SWEEP
        args.bsuite_id = id_or_done
        run.run(args)


if __name__ == "__main__":
    main(Args().parse_args())
