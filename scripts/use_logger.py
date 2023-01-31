from approx.utils.logger import build_logger, get_logger


def main():
    build_logger("test.log")
    get_logger().debug("debug")
    get_logger().info("info")
    get_logger().warning("warning")
    get_logger().error("error")
    get_logger().critical("critical")


if __name__ == '__main__':
    main()
