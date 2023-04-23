#ifndef THREAD_POOL_WRAP_HEADER
#define THREAD_POOL_WRAP_HEADER

#include <BS_thread_pool.hpp>

//Singleton class
typedef BS::thread_pool ThreadPool;

class ThreadPoolWrapper
{
public:
    static ThreadPoolWrapper& getInstance()
    {
        static ThreadPoolWrapper instance{};
        return instance;
    }

private:
    const unsigned int m_threadCount{ std::thread::hardware_concurrency() - 1 };
    ThreadPool m_pool{ std::thread::hardware_concurrency() - 1 };

public:
    unsigned int getThreadCount()
    {
        return m_threadCount;
    }
    BS::thread_pool& getPool()
    {
        return m_pool;
    }

    //Forbidden functions
private:
    ThreadPoolWrapper() {}

public:
    ThreadPoolWrapper(const ThreadPoolWrapper&) = delete;
    void operator=(const ThreadPoolWrapper&) = delete;

};

#endif